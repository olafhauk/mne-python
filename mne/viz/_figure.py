# -*- coding: utf-8 -*-
"""Figure classes for MNE-Python's 2D plots."""

# Authors: Daniel McCloy <dan@mccloy.info>
#
# License: Simplified BSD

from contextlib import contextmanager
import platform
from copy import deepcopy
from itertools import cycle
from functools import partial
from collections import OrderedDict
import numpy as np
from matplotlib.figure import Figure
from .epochs import plot_epochs_image
from .ica import _create_properties_layout
from .utils import (plt_show, plot_sensors, _setup_plot_projector, _events_off,
                    _set_window_title, _merge_annotations, DraggableLine,
                    _get_color_list, logger, _validate_if_list_of_axes,
                    _plot_psd)
from ..defaults import _handle_default
from ..utils import set_config, _check_option, _check_sphere, Bunch
from ..annotations import _sync_onset
from ..time_frequency import psd_welch, psd_multitaper
from ..io.pick import (pick_types, _picks_to_idx, channel_indices_by_type,
                       _DATA_CH_TYPES_SPLIT, _DATA_CH_TYPES_ORDER_DEFAULT,
                       _VALID_CHANNEL_TYPES, _FNIRS_CH_TYPES_SPLIT)


class MNEFigParams:
    """Container object for MNE figure parameters."""

    def __init__(self, **kwargs):
        # default key to close window
        self.close_key = 'escape'
        vars(self).update(**kwargs)


class MNEFigure(Figure):
    """Base class for 2D figures & dialogs; wraps matplotlib.figure.Figure."""

    def __init__(self, **kwargs):
        from matplotlib import rcParams
        # figsize is the only kwarg we pass to matplotlib Figure()
        figsize = kwargs.pop('figsize', None)
        super().__init__(figsize=figsize)
        # things we'll almost always want
        defaults = dict(fgcolor=rcParams['axes.edgecolor'],
                        bgcolor=rcParams['axes.facecolor'])
        for key, value in defaults.items():
            if key not in kwargs:
                kwargs[key] = value
        # add our param object
        self.mne = MNEFigParams(**kwargs)

    def _close(self, event):
        """Handle close events."""
        # remove references from parent fig to child fig
        is_child = getattr(self.mne, 'parent_fig', None) is not None
        is_named = getattr(self.mne, 'fig_name', None) is not None
        if is_child:
            self.mne.parent_fig.mne.child_figs.remove(self)
            if is_named:
                setattr(self.mne.parent_fig.mne, self.mne.fig_name, None)

    def _keypress(self, event):
        """Handle keypress events."""
        if event.key == self.mne.close_key:
            from matplotlib.pyplot import close
            close(self)
        elif event.key == 'f11':  # full screen
            self.canvas.manager.full_screen_toggle()

    def _buttonpress(self, event):
        """Handle buttonpress events."""
        pass

    def _pick(self, event):
        """Handle matplotlib pick events."""
        pass

    def _resize(self, event):
        """Handle window resize events."""
        pass

    def _add_default_callbacks(self, **kwargs):
        """Remove some matplotlib default callbacks and add MNE-Python ones."""
        # Remove matplotlib default keypress catchers
        default_callbacks = list(
            self.canvas.callbacks.callbacks.get('key_press_event', {}))
        for callback in default_callbacks:
            self.canvas.callbacks.disconnect(callback)
        # add our event callbacks
        callbacks = dict(resize_event=self._resize,
                         key_press_event=self._keypress,
                         button_press_event=self._buttonpress,
                         close_event=self._close,
                         pick_event=self._pick)
        callbacks.update(kwargs)
        callback_ids = dict()
        for event, callback in callbacks.items():
            callback_ids[event] = self.canvas.mpl_connect(event, callback)
        # store callback references so they aren't garbage-collected
        self.mne._callback_ids = callback_ids

    def _get_dpi_ratio(self):
        """Get DPI ratio (to handle hi-DPI screens)."""
        dpi_ratio = 1.
        for key in ('_dpi_ratio', '_device_scale'):
            dpi_ratio = getattr(self.canvas, key, dpi_ratio)
        return dpi_ratio

    def _get_size_px(self):
        """Get figure size in pixels."""
        dpi_ratio = self._get_dpi_ratio()
        return self.get_size_inches() * self.dpi / dpi_ratio

    def _inch_to_rel(self, dim_inches, horiz=True):
        """Convert inches to figure-relative distances."""
        fig_w, fig_h = self.get_size_inches()
        w_or_h = fig_w if horiz else fig_h
        return dim_inches / w_or_h


class MNEAnnotationFigure(MNEFigure):
    """Interactive dialog figure for annotations."""

    def _close(self, event):
        """Handle close events (via keypress or window [x])."""
        parent = self.mne.parent_fig
        # disable span selector
        parent.mne.ax_main.selector.active = False
        # clear hover line
        parent._remove_annotation_hover_line()
        # disconnect hover callback
        callback_id = parent.mne._callback_ids['motion_notify_event']
        parent.canvas.callbacks.disconnect(callback_id)
        # do all the other cleanup activities
        super()._close(event)

    def _keypress(self, event):
        """Handle keypress events."""
        text = self.label.get_text()
        key = event.key
        if key == self.mne.close_key:
            from matplotlib.pyplot import close
            close(self)
        elif key == 'backspace':
            text = text[:-1]
        elif key == 'enter':
            self.mne.parent_fig._add_annotation_label(event)
            return
        elif len(key) > 1 or key == ';':  # ignore modifier keys
            return
        else:
            text = text + key
        self.label.set_text(text)
        self.canvas.draw()

    def _radiopress(self, event):
        """Handle Radiobutton clicks for Annotation label selection."""
        # update which button looks active
        buttons = self.mne.radio_ax.buttons
        labels = [label.get_text() for label in buttons.labels]
        idx = labels.index(buttons.value_selected)
        self._set_active_button(idx)
        # update click-drag rectangle color
        color = buttons.circles[idx].get_edgecolor()
        selector = self.mne.parent_fig.mne.ax_main.selector
        selector.rect.set_color(color)
        selector.rectprops.update(dict(facecolor=color))

    def _click_override(self, event):
        """Override MPL radiobutton click detector to use transData."""
        ax = self.mne.radio_ax
        buttons = ax.buttons
        if (buttons.ignore(event) or event.button != 1 or event.inaxes != ax):
            return
        pclicked = ax.transData.inverted().transform((event.x, event.y))
        distances = {}
        for i, (p, t) in enumerate(zip(buttons.circles, buttons.labels)):
            if (t.get_window_extent().contains(event.x, event.y)
                    or np.linalg.norm(pclicked - p.center) < p.radius):
                distances[i] = np.linalg.norm(pclicked - p.center)
        if len(distances) > 0:
            closest = min(distances, key=distances.get)
            buttons.set_active(closest)

    def _set_active_button(self, idx):
        """Set active button in annotation dialog figure."""
        buttons = self.mne.radio_ax.buttons
        with _events_off(buttons):
            buttons.set_active(idx)
        for circle in buttons.circles:
            circle.set_facecolor(self.mne.parent_fig.mne.bgcolor)
        # active circle gets filled in, partially transparent
        color = list(buttons.circles[idx].get_edgecolor())
        color[-1] = 0.5
        buttons.circles[idx].set_facecolor(color)
        self.canvas.draw()


class MNESelectionFigure(MNEFigure):
    """Interactive dialog figure for channel selections."""

    def _close(self, event):
        """Handle close events."""
        from matplotlib.pyplot import close
        self.mne.parent_fig.mne.child_figs.remove(self)
        self.mne.fig_selection = None
        # selection fig & main fig tightly integrated; closing one closes both
        close(self.mne.parent_fig)

    def _keypress(self, event):
        """Handle keypress events."""
        if event.key in ('up', 'down', 'b'):
            self.mne.parent_fig._keypress(event)
        else:  # check for close key
            super()._keypress(event)

    def _radiopress(self, event):
        """Handle RadioButton clicks for channel selection groups."""
        selections_dict = self.mne.parent_fig.mne.ch_selections
        buttons = self.mne.radio_ax.buttons
        labels = [label.get_text() for label in buttons.labels]
        this_label = buttons.value_selected
        parent = self.mne.parent_fig
        if this_label == 'Custom' and not len(selections_dict['Custom']):
            with _events_off(buttons):
                buttons.set_active(self.mne.old_selection)
            return
        # clicking a selection cancels butterfly mode
        if parent.mne.butterfly:
            parent._toggle_butterfly()
            with _events_off(buttons):
                buttons.set_active(labels.index(this_label))
        parent._update_selection()

    def _set_custom_selection(self):
        """Set custom selection by lasso selector."""
        chs = self.lasso.selection
        parent = self.mne.parent_fig
        buttons = self.mne.radio_ax.buttons
        if not len(chs):
            return
        labels = [label.get_text() for label in buttons.labels]
        inds = np.in1d(parent.mne.ch_names, chs)
        parent.mne.ch_selections['Custom'] = inds.nonzero()[0]
        buttons.set_active(labels.index('Custom'))

    def _style_radio_buttons_butterfly(self):
        """Handle RadioButton state for keyboard interactions."""
        # Show all radio buttons as selected when in butterfly mode
        parent = self.mne.parent_fig
        buttons = self.mne.radio_ax.buttons
        color = (buttons.activecolor if parent.mne.butterfly else
                 parent.mne.bgcolor)
        for circle in buttons.circles:
            circle.set_facecolor(color)
        # when leaving butterfly mode, make most-recently-used selection active
        if not parent.mne.butterfly:
            with _events_off(buttons):
                buttons.set_active(self.mne.old_selection)
        # update the sensors too
        parent._update_highlighted_sensors()


class MNEBrowseFigure(MNEFigure):
    """Interactive figure with scrollbars, for data browsing."""

    def __init__(self, inst, figsize, ica=None, xlabel='Time (s)', **kwargs):
        from matplotlib.colors import to_rgba_array
        from matplotlib.ticker import (FixedLocator, FixedFormatter,
                                       FuncFormatter, NullFormatter)
        from matplotlib.patches import Rectangle
        from matplotlib.widgets import Button
        from matplotlib.transforms import blended_transform_factory
        from mpl_toolkits.axes_grid1.axes_size import Fixed
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        from .. import BaseEpochs
        from ..io import BaseRaw
        from ..preprocessing import ICA

        super().__init__(figsize=figsize, inst=inst, ica=ica, **kwargs)

        # what kind of data are we dealing with?
        if isinstance(ica, ICA):
            self.mne.instance_type = 'ica'
        elif isinstance(inst, BaseRaw):
            self.mne.instance_type = 'raw'
        elif isinstance(inst, BaseEpochs):
            self.mne.instance_type = 'epochs'
        else:
            raise TypeError('Expected an instance of Raw, Epochs, or ICA, '
                            f'got {type(inst)}.')
        self.mne.ica_type = None
        if self.mne.instance_type == 'ica':
            if isinstance(self.mne.ica_inst, BaseRaw):
                self.mne.ica_type = 'raw'
            elif isinstance(self.mne.ica_inst, BaseEpochs):
                self.mne.ica_type = 'epochs'
        self.mne.is_epochs = 'epochs' in (self.mne.instance_type,
                                          self.mne.ica_type)

        # things that always start the same
        self.mne.ch_start = 0
        self.mne.projector = None
        self.mne.projs_active = np.array([p['active'] for p in self.mne.projs])
        self.mne.whitened_ch_names = list()
        self.mne.use_noise_cov = self.mne.noise_cov is not None
        self.mne.zorder = dict(patch=0, grid=1, ann=2, events=3, bads=4,
                               data=5, mag=6, grad=7, scalebar=8, vline=9)
        # additional params for epochs (won't affect raw / ICA)
        self.mne.epoch_traces = list()
        self.mne.bad_epochs = list()
        # annotations
        self.mne.annotations = list()
        self.mne.hscroll_annotations = list()
        self.mne.annotation_segments = list()
        self.mne.annotation_texts = list()
        self.mne.new_annotation_labels = list()
        self.mne.annotation_segment_colors = dict()
        self.mne.annotation_hover_line = None
        self.mne.draggable_annotations = False
        # lines
        self.mne.event_lines = None
        self.mne.event_texts = list()
        self.mne.vline_visible = False
        # scalings
        self.mne.scale_factor = 0.5 if self.mne.butterfly else 1.
        self.mne.scalebars = dict()
        self.mne.scalebar_texts = dict()
        # ancillary child figures
        self.mne.child_figs = list()
        self.mne.fig_help = None
        self.mne.fig_proj = None
        self.mne.fig_histogram = None
        self.mne.fig_selection = None
        self.mne.fig_annotation = None

        # MAIN AXES: default sizes (inches)
        # XXX simpler with constrained_layout? (when it's no longer "beta")
        l_margin = 1.
        r_margin = 0.1
        b_margin = 0.45
        t_margin = 0.25
        scroll_width = 0.25
        hscroll_dist = 0.25
        vscroll_dist = 0.1
        help_width = scroll_width * 2
        # MAIN AXES: default margins (figure-relative coordinates)
        left = self._inch_to_rel(l_margin - vscroll_dist - help_width)
        right = 1 - self._inch_to_rel(r_margin)
        bottom = self._inch_to_rel(b_margin, horiz=False)
        top = 1 - self._inch_to_rel(t_margin, horiz=False)
        width = right - left
        height = top - bottom
        position = [left, bottom, width, height]
        # Main axes must be a subplot for subplots_adjust to work (so user can
        # adjust margins). That's why we don't use the Divider class directly.
        ax_main = self.add_subplot(1, 1, 1, position=position)
        self.subplotpars.update(left=left, bottom=bottom, top=top, right=right)
        div = make_axes_locatable(ax_main)
        # this only gets shown in zen mode
        self.mne.zen_xlabel = ax_main.set_xlabel(xlabel)
        self.mne.zen_xlabel.set_visible(not self.mne.scrollbars_visible)

        # SCROLLBARS
        ax_hscroll = div.append_axes(position='bottom',
                                     size=Fixed(scroll_width),
                                     pad=Fixed(hscroll_dist))
        ax_vscroll = div.append_axes(position='right',
                                     size=Fixed(scroll_width),
                                     pad=Fixed(vscroll_dist))
        ax_hscroll.get_yaxis().set_visible(False)
        ax_hscroll.set_xlabel(xlabel)
        ax_vscroll.set_axis_off()
        # HORIZONTAL SCROLLBAR PATCHES (FOR MARKING BAD EPOCHS)
        if self.mne.is_epochs:
            epoch_nums = self.mne.inst.selection
            for ix, _ in enumerate(epoch_nums):
                start = self.mne.boundary_times[ix]
                width = np.diff(self.mne.boundary_times[ix:ix + 2])[0]
                ax_hscroll.add_patch(
                    Rectangle((start, 0), width, 1, color='none',
                              zorder=self.mne.zorder['patch']))
            # add epoch boundaries & center epoch numbers between boundaries
            midpoints = np.convolve(self.mne.boundary_times, np.ones(2),
                                    mode='valid') / 2
            # both axes, major ticks: gridlines
            for _ax in (ax_main, ax_hscroll):
                _ax.xaxis.set_major_locator(
                    FixedLocator(self.mne.boundary_times[1:-1]))
                _ax.xaxis.set_major_formatter(NullFormatter())
            grid_kwargs = dict(color=self.mne.fgcolor, axis='x',
                               zorder=self.mne.zorder['grid'])
            ax_main.grid(linewidth=2, linestyle='dashed', **grid_kwargs)
            ax_hscroll.grid(alpha=0.5, linewidth=0.5, linestyle='solid',
                            **grid_kwargs)
            # main axes, minor ticks: ticklabel (epoch number) for every epoch
            ax_main.xaxis.set_minor_locator(FixedLocator(midpoints))
            ax_main.xaxis.set_minor_formatter(FixedFormatter(epoch_nums))
            # hscroll axes, minor ticks: up to 20 ticklabels (epoch numbers)
            ax_hscroll.xaxis.set_minor_locator(
                FixedLocator(midpoints, nbins=20))
            ax_hscroll.xaxis.set_minor_formatter(
                FuncFormatter(lambda x, pos: self._get_epoch_num_from_time(x)))
            # hide some ticks
            ax_main.tick_params(axis='x', which='major', bottom=False)
            ax_hscroll.tick_params(axis='x', which='both', bottom=False)

        # VERTICAL SCROLLBAR PATCHES (COLORED BY CHANNEL TYPE)
        ch_order = self.mne.ch_order
        for ix, pick in enumerate(ch_order):
            this_color = (self.mne.ch_color_bad
                          if self.mne.ch_names[pick] in self.mne.info['bads']
                          else self.mne.ch_color_dict)
            if isinstance(this_color, dict):
                this_color = this_color[self.mne.ch_types[pick]]
            ax_vscroll.add_patch(
                Rectangle((0, ix), 1, 1, color=this_color,
                          zorder=self.mne.zorder['patch']))
        ax_vscroll.set_ylim(len(ch_order), 0)
        ax_vscroll.set_visible(not self.mne.butterfly)
        # SCROLLBAR VISIBLE SELECTION PATCHES
        sel_kwargs = dict(alpha=0.3, linewidth=4, clip_on=False,
                          edgecolor=self.mne.fgcolor)
        vsel_patch = Rectangle((0, 0), 1, self.mne.n_channels,
                               facecolor=self.mne.bgcolor, **sel_kwargs)
        ax_vscroll.add_patch(vsel_patch)
        hsel_facecolor = np.average(
            np.vstack((to_rgba_array(self.mne.fgcolor),
                       to_rgba_array(self.mne.bgcolor))),
            axis=0, weights=(3, 1))  # 75% foreground, 25% background
        hsel_patch = Rectangle((self.mne.t_start, 0), self.mne.duration, 1,
                               facecolor=hsel_facecolor, **sel_kwargs)
        ax_hscroll.add_patch(hsel_patch)
        ax_hscroll.set_xlim(self.mne.first_time, self.mne.first_time +
                            self.mne.n_times / self.mne.info['sfreq'])
        # VLINE
        vline_color = (0., 0.75, 0.)
        vline_kwargs = dict(visible=False, animated=True,
                            zorder=self.mne.zorder['vline'])
        if self.mne.is_epochs:
            x = np.arange(self.mne.n_epochs)
            vline = ax_main.vlines(
                x, 0, 1, colors=vline_color, **vline_kwargs)
            vline.set_transform(blended_transform_factory(ax_main.transData,
                                                          ax_main.transAxes))
            vline_hscroll = None
        else:
            vline = ax_main.axvline(0, color=vline_color, **vline_kwargs)
            vline_hscroll = ax_hscroll.axvline(0, color=vline_color,
                                               **vline_kwargs)
        vline_text = ax_hscroll.text(
            self.mne.first_time, 1.2, '', fontsize=10, ha='right', va='bottom',
            color=vline_color, **vline_kwargs)

        # HELP BUTTON: initialize in the wrong spot...
        ax_help = div.append_axes(position='left',
                                  size=Fixed(help_width),
                                  pad=Fixed(vscroll_dist))
        # HELP BUTTON: ...move it down by changing its locator
        loc = div.new_locator(nx=0, ny=0)
        ax_help.set_axes_locator(loc)
        # HELP BUTTON: make it a proper button
        with _patched_canvas(ax_help.figure):
            self.mne.button_help = Button(ax_help, 'Help')
        # PROJ BUTTON
        ax_proj = None
        if len(self.mne.projs) and not inst.proj:
            proj_button_pos = [
                1 - self._inch_to_rel(r_margin + scroll_width),  # left
                self._inch_to_rel(b_margin, horiz=False),        # bottom
                self._inch_to_rel(scroll_width),                 # width
                self._inch_to_rel(scroll_width, horiz=False)     # height
            ]
            loc = div.new_locator(nx=4, ny=0)
            ax_proj = self.add_axes(proj_button_pos)
            ax_proj.set_axes_locator(loc)
            with _patched_canvas(ax_help.figure):
                self.mne.button_proj = Button(ax_proj, 'Prj')

        # INIT TRACES
        self.mne.trace_kwargs = dict(antialiased=True, linewidth=0.5)
        self.mne.traces = ax_main.plot(
            np.full((1, self.mne.n_channels), np.nan), **self.mne.trace_kwargs)

        # SAVE UI ELEMENT HANDLES
        vars(self.mne).update(
            ax_main=ax_main, ax_help=ax_help, ax_proj=ax_proj,
            ax_hscroll=ax_hscroll, ax_vscroll=ax_vscroll,
            vsel_patch=vsel_patch, hsel_patch=hsel_patch, vline=vline,
            vline_hscroll=vline_hscroll, vline_text=vline_text)

    def _close(self, event):
        """Handle close events (via keypress or window [x])."""
        from matplotlib.pyplot import close
        # write out bad epochs (after converting epoch numbers to indices)
        if self.mne.instance_type == 'epochs':
            bad_ixs = np.in1d(self.mne.inst.selection,
                              self.mne.bad_epochs).nonzero()[0]
            self.mne.inst.drop(bad_ixs)
        # write bad channels back to instance (don't do this for proj;
        # proj checkboxes are for viz only and shouldn't modify the instance)
        if self.mne.instance_type in ('raw', 'epochs'):
            self.mne.inst.info['bads'] = self.mne.info['bads']
            logger.info(
                f"Channels marked as bad: {self.mne.info['bads'] or 'none'}")
        # ICA excludes
        elif self.mne.instance_type == 'ica':
            self.mne.ica.exclude = [self.mne.ica._ica_names.index(ch)
                                    for ch in self.mne.info['bads']]
        # write window size to config
        size = ','.join(self.get_size_inches().astype(str))
        set_config('MNE_BROWSE_RAW_SIZE', size, set_env=False)
        # Clean up child figures (don't pop(), child figs remove themselves)
        while len(self.mne.child_figs):
            fig = self.mne.child_figs[-1]
            close(fig)

    def _resize(self, event):
        """Handle resize event for mne_browse-style plots (Raw/Epochs/ICA)."""
        old_width, old_height = self.mne.fig_size_px
        new_width, new_height = self._get_size_px()
        new_margins = _calc_new_margins(
            self, old_width, old_height, new_width, new_height)
        self.subplots_adjust(**new_margins)
        # zen mode bookkeeping
        self.mne.zen_w *= old_width / new_width
        self.mne.zen_h *= old_height / new_height
        self.mne.fig_size_px = (new_width, new_height)
        # for blitting
        self.canvas.draw_idle()
        self.canvas.flush_events()
        self.mne.bg = self.canvas.copy_from_bbox(self.bbox)

    def _hover(self, event):
        """Handle motion event when annotating."""
        if (event.button is not None or event.xdata is None or
                event.inaxes != self.mne.ax_main):
            return
        if not self.mne.draggable_annotations:
            self._remove_annotation_hover_line()
            return
        from matplotlib.patheffects import Stroke, Normal
        for coll in self.mne.annotations:
            if coll.contains(event)[0]:
                path = coll.get_paths()
                assert len(path) == 1
                path = path[0]
                color = coll.get_edgecolors()[0]
                ylim = self.mne.ax_main.get_ylim()
                # are we on the left or right edge?
                _l = path.vertices[:, 0].min()
                _r = path.vertices[:, 0].max()
                x = _l if abs(event.xdata - _l) < abs(event.xdata - _r) else _r
                mask = path.vertices[:, 0] == x

                def drag_callback(x0):
                    path.vertices[mask, 0] = x0

                # create or update the DraggableLine
                hover_line = self.mne.annotation_hover_line
                if hover_line is None:
                    line = self.mne.ax_main.plot([x, x], ylim, color=color,
                                                 linewidth=2, pickradius=5.)[0]
                    hover_line = DraggableLine(
                        line, self._modify_annotation, drag_callback)
                else:
                    hover_line.set_x(x)
                    hover_line.drag_callback = drag_callback
                # style the line
                line = hover_line.line
                patheff = [Stroke(linewidth=4, foreground=color, alpha=0.5),
                           Normal()]
                line.set_path_effects(patheff if line.contains(event)[0] else
                                      patheff[1:])
                self.mne.ax_main.selector.active = False
                self.mne.annotation_hover_line = hover_line
                self.canvas.draw_idle()
                return
        self._remove_annotation_hover_line()

    def _keypress(self, event):
        """Handle keypress events."""
        key = event.key
        n_channels = self.mne.n_channels
        if self.mne.is_epochs:
            last_time = self.mne.n_times / self.mne.info['sfreq']
        else:
            last_time = self.mne.inst.times[-1]
        # scroll up/down
        if key in ('down', 'up'):
            direction = -1 if key == 'up' else 1
            # butterfly case
            if self.mne.butterfly:
                return
            # group_by case
            elif self.mne.fig_selection is not None:
                buttons = self.mne.fig_selection.mne.radio_ax.buttons
                labels = [label.get_text() for label in buttons.labels]
                current_label = buttons.value_selected
                current_idx = labels.index(current_label)
                selections_dict = self.mne.ch_selections
                penult = current_idx < (len(labels) - 1)
                pre_penult = current_idx < (len(labels) - 2)
                has_custom = selections_dict.get('Custom', None) is not None
                def_custom = len(selections_dict.get('Custom', list()))
                up_ok = key == 'up' and current_idx > 0
                down_ok = key == 'down' and (
                    pre_penult or
                    (penult and not has_custom) or
                    (penult and has_custom and def_custom))
                if up_ok or down_ok:
                    buttons.set_active(current_idx + direction)
            # normal case
            else:
                ceiling = len(self.mne.ch_order) - n_channels
                ch_start = self.mne.ch_start + direction * n_channels
                self.mne.ch_start = np.clip(ch_start, 0, ceiling)
                self._update_picks()
                self._update_vscroll()
                self._redraw()
        # scroll left/right
        elif key in ('right', 'left', 'shift+right', 'shift+left'):
            old_t_start = self.mne.t_start
            direction = 1 if key.endswith('right') else -1
            if self.mne.is_epochs:
                denom = 1 if key.startswith('shift') else self.mne.n_epochs
            else:
                denom = 1 if key.startswith('shift') else 4
            t_max = last_time - self.mne.duration
            t_start = self.mne.t_start + direction * self.mne.duration / denom
            self.mne.t_start = np.clip(t_start, self.mne.first_time, t_max)
            if self.mne.t_start != old_t_start:
                self._update_hscroll()
                self._redraw(annotations=True)
        # scale traces
        elif key in ('=', '+', '-'):
            scaler = 1 / 1.1 if key == '-' else 1.1
            self.mne.scale_factor *= scaler
            self._redraw(update_data=False)
        # change number of visible channels
        elif (key in ('pageup', 'pagedown') and
              self.mne.fig_selection is None and
              not self.mne.butterfly):
            new_n_ch = n_channels + (1 if key == 'pageup' else -1)
            self.mne.n_channels = np.clip(new_n_ch, 1, len(self.mne.ch_order))
            # add new chs from above if we're at the bottom of the scrollbar
            ch_end = self.mne.ch_start + self.mne.n_channels
            if ch_end > len(self.mne.ch_order) and self.mne.ch_start > 0:
                self.mne.ch_start -= 1
                self._update_vscroll()
            # redraw only if changed
            if self.mne.n_channels != n_channels:
                self._update_picks()
                self._update_trace_offsets()
                self._redraw(annotations=True)
        # change duration
        elif key in ('home', 'end'):
            dur_delta = 1 if key == 'end' else -1
            if self.mne.is_epochs:
                self.mne.n_epochs = np.clip(self.mne.n_epochs + dur_delta,
                                            1, len(self.mne.inst))
                min_dur = len(self.mne.inst.times) / self.mne.info['sfreq']
                dur_delta *= min_dur
            else:
                min_dur = 3 * np.diff(self.mne.inst.times[:2])[0]
            old_dur = self.mne.duration
            new_dur = self.mne.duration + dur_delta
            self.mne.duration = np.clip(new_dur, min_dur, last_time)
            if self.mne.duration != old_dur:
                if self.mne.t_start + self.mne.duration > last_time:
                    self.mne.t_start = last_time - self.mne.duration
                self._update_hscroll()
                if key == 'end' and self.mne.vline_visible:  # prevent flicker
                    self._show_vline(None)
                self._redraw()
        elif key == '?':  # help window
            self._toggle_help_fig(event)
        elif key == 'a':  # annotation mode
            self._toggle_annotation_fig()
        elif key == 'b' and self.mne.instance_type != 'ica':  # butterfly mode
            self._toggle_butterfly()
        elif key == 'd':  # DC shift
            self.mne.remove_dc = not self.mne.remove_dc
            self._redraw()
        elif key == 'h' and self.mne.instance_type == 'epochs':  # histogram
            self._toggle_epoch_histogram()
        elif key == 'j' and len(self.mne.projs):  # SSP window
            self._toggle_proj_fig()
        elif key == 'p':  # toggle draggable annotations
            self._toggle_draggable_annotations(event)
            if self.mne.fig_annotation is not None:
                checkbox = self.mne.fig_annotation.mne.drag_checkbox
                with _events_off(checkbox):
                    checkbox.set_active(0)
        elif key == 's':  # scalebars
            self._toggle_scalebars(event)
        elif key == 'w':  # toggle noise cov whitening
            if self.mne.noise_cov is not None:
                self.mne.use_noise_cov = not self.mne.use_noise_cov
                self._update_projector()
                self._update_yaxis_labels()  # add/remove italics
                self._redraw()
        elif key == 'z':  # zen mode: hide scrollbars and buttons
            self._toggle_scrollbars()
            self._redraw(update_data=False)
        else:  # check for close key / fullscreen toggle
            super()._keypress(event)

    def _buttonpress(self, event):
        """Handle mouse clicks."""
        butterfly = self.mne.butterfly
        annotating = self.mne.fig_annotation is not None
        ax_main = self.mne.ax_main
        inst = self.mne.inst
        # ignore middle clicks, scroll wheel events, and clicks outside axes
        if event.button not in (1, 3) or event.inaxes is None:
            return
        elif event.button == 1:  # left-click (primary)
            # click in main axes
            if (event.inaxes == ax_main and not annotating):
                if self.mne.instance_type == 'epochs' or not butterfly:
                    for line in self.mne.traces + self.mne.epoch_traces:
                        if line.contains(event)[0]:
                            if self.mne.instance_type == 'epochs':
                                self._toggle_bad_epoch(event)
                            else:
                                idx = self.mne.traces.index(line)
                                self._toggle_bad_channel(idx)
                            return
                self._show_vline(event.xdata)  # butterfly / not on data trace
                return
            # click in vertical scrollbar
            elif event.inaxes == self.mne.ax_vscroll:
                if self.mne.fig_selection is not None:
                    self._change_selection_vscroll(event)
                elif self._check_update_vscroll_clicked(event):
                    self._redraw()
            # click in horizontal scrollbar
            elif event.inaxes == self.mne.ax_hscroll:
                if self._check_update_hscroll_clicked(event):
                    self._redraw()
            # click on proj button
            elif event.inaxes == self.mne.ax_proj:
                self._toggle_proj_fig(event)
            # click on help button
            elif event.inaxes == self.mne.ax_help:
                self._toggle_help_fig(event)
        else:  # right-click (secondary)
            if annotating:
                if any(c.contains(event)[0] for c in ax_main.collections):
                    xdata = event.xdata - self.mne.first_time
                    start = _sync_onset(inst, inst.annotations.onset)
                    end = start + inst.annotations.duration
                    ann_idx = np.where((xdata > start) & (xdata < end))[0]
                    inst.annotations.delete(ann_idx)  # only first one deleted
                self._remove_annotation_hover_line()
                self._draw_annotations()
                self.canvas.draw_idle()
            elif event.inaxes == ax_main:  # hide green line
                self._blit_vline(False)

    def _pick(self, event):
        """Handle matplotlib pick events."""
        from matplotlib.text import Text
        if self.mne.butterfly:
            return
        # clicked on channel name
        if isinstance(event.artist, Text):
            ch_name = event.artist.get_text()
            ind = self.mne.ch_names[self.mne.picks].tolist().index(ch_name)
            if event.mouseevent.button == 1:  # left click
                self._toggle_bad_channel(ind)
            elif event.mouseevent.button == 3:  # right click
                self._create_ch_context_fig(ind)

    def _new_child_figure(self, fig_name, **kwargs):
        """Instantiate a new MNE dialog figure (with event listeners)."""
        fig = _figure(toolbar=False, parent_fig=self, fig_name=fig_name,
                      **kwargs)
        fig._add_default_callbacks()
        self.mne.child_figs.append(fig)
        if isinstance(fig_name, str):
            setattr(self.mne, fig_name, fig)
        return fig

    def _create_ch_context_fig(self, idx):
        """Show context figure; idx is index of **visible** channels."""
        inst = self.mne.instance_type
        pick = self.mne.picks[idx]
        if inst == 'raw':
            self._create_ch_location_fig(pick)
        elif inst == 'ica':
            self._create_ica_properties_fig(pick)
        else:
            self._create_epoch_image_fig(pick)

    def _create_ch_location_fig(self, pick):
        """Show channel location figure."""
        from .utils import _channel_type_prettyprint
        ch_name = self.mne.ch_names[pick]
        ch_type = self.mne.ch_types[pick]
        if ch_type not in _DATA_CH_TYPES_SPLIT:
            return
        # create figure and axes
        fig = self._new_child_figure(figsize=(4, 4), fig_name=None,
                                     window_title=f'Location of {ch_name}')
        ax = fig.add_subplot(111)
        title = f'{ch_name} position ({_channel_type_prettyprint[ch_type]})'
        _ = plot_sensors(self.mne.info, ch_type=ch_type, axes=ax,
                         title=title, kind='select')
        # highlight desired channel & disable interactivity
        inds = np.in1d(fig.lasso.ch_names, [ch_name])
        fig.lasso.disconnect()
        fig.lasso.alpha_other = 0.3
        fig.lasso.linewidth_selected = 3
        fig.lasso.style_sensors(inds)
        plt_show(fig=fig)

    def _create_ica_properties_fig(self, pick):
        """Show ICA properties for the selected component."""
        ch_name = self.mne.ch_names[pick]
        fig = self._new_child_figure(figsize=(7, 6), fig_name=None,
                                     window_title=f'{ch_name} properties')
        fig, axes = _create_properties_layout(fig=fig)
        self.mne.ica.plot_properties(self.mne.ica_inst, picks=pick, axes=axes)

    def _create_epoch_image_fig(self, pick):
        """Show epochs image for the selected channel."""
        from matplotlib.gridspec import GridSpec
        ch_name = self.mne.ch_names[pick]
        fig = self._new_child_figure(figsize=(6, 4), fig_name=None,
                                     window_title=f'Epochs image ({ch_name})')
        gs = GridSpec(nrows=3, ncols=10)
        fig.add_subplot(gs[:2, :9])
        fig.add_subplot(gs[2, :9])
        fig.add_subplot(gs[:2, 9])
        plot_epochs_image(self.mne.inst, picks=pick, fig=fig)

    def _toggle_epoch_histogram(self):
        """Show or hide peak-to-peak histogram of channel amplitudes."""
        if self.mne.fig_histogram is None:
            self._create_epoch_histogram()
            plt_show(fig=self.mne.fig_histogram)
        else:
            from matplotlib.pyplot import close
            close(self.mne.fig_histogram)

    def _create_epoch_histogram(self):
        """Create peak-to-peak histogram of channel amplitudes."""
        epochs = self.mne.inst
        data = OrderedDict()
        ptp = np.ptp(epochs.get_data(), axis=2)
        for ch_type in ('eeg', 'mag', 'grad'):
            if ch_type in epochs:
                data[ch_type] = ptp.T[self.mne.ch_types == ch_type].ravel()
        units = _handle_default('units')
        titles = _handle_default('titles')
        colors = _handle_default('color')
        scalings = _handle_default('scalings')
        title = 'Histogram of peak-to-peak amplitudes'
        figsize = (4, 1 + 1.5 * len(data))
        fig = self._new_child_figure(figsize=figsize, fig_name='fig_histogram',
                                     window_title=title)
        for ix, (_ch_type, _data) in enumerate(data.items()):
            ax = fig.add_subplot(len(data), 1, ix + 1)
            ax.set(title=titles[_ch_type], xlabel=units[_ch_type],
                   ylabel='Count')
            # set histogram bin range based on rejection thresholds
            reject = None
            _range = None
            if epochs.reject is not None and _ch_type in epochs.reject:
                reject = epochs.reject[_ch_type] * scalings[_ch_type]
                _range = (0., reject * 1.1)
            # plot it
            ax.hist(_data * scalings[_ch_type], bins=100,
                    color=colors[_ch_type], range=_range)
            if reject is not None:
                ax.plot((reject, reject), (0, ax.get_ylim()[1]), color='r')
        # finalize
        fig.suptitle(title, y=0.99)
        kwargs = dict(bottom=fig._inch_to_rel(0.5, horiz=False),
                      top=1 - fig._inch_to_rel(0.5, horiz=False),
                      left=fig._inch_to_rel(0.75),
                      right=1 - fig._inch_to_rel(0.25))
        fig.subplots_adjust(hspace=0.7, **kwargs)
        self.mne.fig_histogram = fig
        plt_show(fig=fig)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # HELP DIALOG
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _create_help_fig(self):
        """Create help dialog window."""
        text = {key: val for key, val in self._get_help_text().items()
                if val is not None}
        keys = ''
        vals = ''
        for key, val in text.items():
            newsection = '\n' if key.startswith('_') else ''
            key = key[1:] if key.startswith('_') else key
            newlines = '\n' * len(val.split('\n'))  # handle multiline values
            keys += f'{newsection}{key}      {newlines}'
            vals += f'{newsection}{val}\n'
        # calc figure size
        n_lines = len(keys.split('\n'))
        longest_key = max(len(k) for k in text.keys())
        longest_val = max(max(len(w) for w in v.split('\n')) if '\n' in v else
                          len(v) for v in text.values())
        width = (longest_key + longest_val) / 12
        height = (n_lines) / 5
        # create figure and axes
        fig = self._new_child_figure(figsize=(width, height),
                                     fig_name='fig_help',
                                     window_title='Help')
        ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
        ax.set_axis_off()
        kwargs = dict(va='top', linespacing=1.5, usetex=False)
        ax.text(0.42, 1, keys, ma='right', ha='right', **kwargs)
        ax.text(0.42, 1, vals, ma='left', ha='left', **kwargs)

    def _toggle_help_fig(self, event):
        """Show/hide the help dialog window."""
        if self.mne.fig_help is None:
            self._create_help_fig()
            plt_show(fig=self.mne.fig_help)
        else:
            from matplotlib.pyplot import close
            close(self.mne.fig_help)

    def _get_help_text(self):
        """Generate help dialog text; `None`-valued entries removed later."""
        inst = self.mne.instance_type
        is_raw = inst == 'raw'
        is_epo = inst == 'epochs'
        is_ica = inst == 'ica'
        has_proj = bool(len(self.mne.projs))
        # adapt keys to different platforms
        is_mac = platform.system() == 'Darwin'
        dur_keys = ('⌘ + ←', '⌘ + →') if is_mac else ('Home', 'End')
        ch_keys = ('⌘ + ↑', '⌘ + ↓') if is_mac else ('Page up', 'Page down')
        # adapt descriptions to different instance types
        ch_cmp = 'component' if is_ica else 'channel'
        ch_epo = 'epoch' if is_epo else 'channel'
        ica_bad = 'Mark/unmark component for exclusion'
        dur_vals = ([f'Show {n} epochs' for n in ('fewer', 'more')]
                    if self.mne.is_epochs else
                    [f'Show {d} time window' for d in ('shorter', 'longer')])
        ch_vals = [f'{inc_dec} number of visible {ch_cmp}s' for inc_dec in
                   ('Increase', 'Decrease')]
        lclick_data = ica_bad if is_ica else f'Mark/unmark bad {ch_epo}'
        lclick_name = (ica_bad if is_ica else 'Mark/unmark bad channel')
        rclick_name = dict(ica='Show diagnostics for component',
                           epochs='Show imageplot for channel',
                           raw='Show channel location')[inst]
        # TODO not yet implemented
        # ldrag = ('Show spectrum plot for selected time span;\nor (in '
        #          'annotation mode) add annotation') if inst== 'raw' else None
        ldrag = 'add annotation (in annotation mode)' if is_raw else None
        noise_cov = (None if self.mne.noise_cov is None else
                     'Toggle signal whitening')
        scrl = '1 epoch' if self.mne.is_epochs else '¼ window'
        # below, value " " is a hack to make "\n".split(value) have length 1
        help_text = OrderedDict([
            ('_NAVIGATION', ' '),
            ('→', f'Scroll {scrl} right (scroll full window with Shift + →)'),
            ('←', f'Scroll {scrl} left (scroll full window with Shift + ←)'),
            (dur_keys[0], dur_vals[0]),
            (dur_keys[1], dur_vals[1]),
            ('↑', f'Scroll up ({ch_cmp}s)'),
            ('↓', f'Scroll down ({ch_cmp}s)'),
            (ch_keys[0], ch_vals[0]),
            (ch_keys[1], ch_vals[1]),
            ('_SIGNAL TRANSFORMATIONS', ' '),
            ('+ or =', 'Increase signal scaling'),
            ('-', 'Decrease signal scaling'),
            ('b', 'Toggle butterfly mode' if not is_ica else None),
            ('d', 'Toggle DC removal' if is_raw else None),
            ('w', noise_cov),
            ('_USER INTERFACE', ' '),
            ('a', 'Toggle annotation mode' if is_raw else None),
            ('h', 'Toggle peak-to-peak histogram' if is_epo else None),
            ('j', 'Toggle SSP projector window' if has_proj else None),
            ('p', 'Toggle draggable annotations' if is_raw else None),
            ('s', 'Toggle scalebars' if not is_ica else None),
            ('z', 'Toggle scrollbars'),
            ('F11', 'Toggle fullscreen'),
            ('?', 'Open this help window'),
            ('esc', 'Close focused figure or dialog window'),
            ('_MOUSE INTERACTION', ' '),
            (f'Left-click {ch_cmp} name', lclick_name),
            (f'Left-click {ch_cmp} data', lclick_data),
            ('Left-click-and-drag on plot', ldrag),
            ('Left-click on plot background', 'Place vertical guide'),
            ('Right-click on plot background', 'Clear vertical guide'),
            ('Right-click on channel name', rclick_name)
        ])
        return help_text

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # ANNOTATIONS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _create_annotation_fig(self):
        """Create the annotation dialog window."""
        from matplotlib.widgets import Button, SpanSelector, CheckButtons
        from mpl_toolkits.axes_grid1.axes_size import Fixed
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
        # make figure
        labels = np.array(sorted(set(self.mne.inst.annotations.description)))
        width, var_height, fixed_height, pad = \
            self._compute_annotation_figsize(len(labels))
        figsize = (width, var_height + fixed_height)
        fig = self._new_child_figure(figsize=figsize,
                                     FigureClass=MNEAnnotationFigure,
                                     fig_name='fig_annotation',
                                     window_title='Annotations')
        # make main axes
        left = fig._inch_to_rel(pad)
        bottom = fig._inch_to_rel(pad, horiz=False)
        width = 1 - 2 * left
        height = 1 - 2 * bottom
        fig.mne.radio_ax = fig.add_axes((left, bottom, width, height),
                                        frame_on=False, aspect='equal')
        div = make_axes_locatable(fig.mne.radio_ax)
        self._update_annotation_fig()  # populate w/ radio buttons & labels
        # append instructions at top
        instructions_ax = div.append_axes(position='top', size=Fixed(1),
                                          pad=Fixed(5 * pad))
        # XXX when we support a newer matplotlib (something >3.0) the
        # instructions can have inline bold formatting:
        # instructions = '\n'.join(
        #     [r'$\mathbf{Left‐click~&~drag~on~plot:}$ create/modify annotation',  # noqa E501
        #      r'$\mathbf{Right‐click~on~plot~annotation:}$ delete annotation',
        #      r'$\mathbf{Type~in~annotation~window:}$ modify new label name',
        #      r'$\mathbf{Enter~(or~click~button):}$ add new label to list',
        #      r'$\mathbf{Esc:}$ exit annotation mode & close window'])
        instructions = '\n'.join(
            ['Left click & drag on plot: create/modify annotation',
             'Right click on plot annotation: delete annotation',
             'Type in annotation window: modify new label name',
             'Enter (or click button): add new label to list',
             'Esc: exit annotation mode & close window'])
        instructions_ax.text(0, 1, instructions, va='top', ha='left',
                             usetex=False)  # force use of MPL mathtext parser
        instructions_ax.set_axis_off()
        # append text entry axes at bottom
        text_entry_ax = div.append_axes(position='bottom', size=Fixed(3 * pad),
                                        pad=Fixed(pad))
        text_entry_ax.text(0.4, 0.5, 'New label:', va='center', ha='right',
                           weight='bold')
        fig.label = text_entry_ax.text(0.5, 0.5, 'BAD_', va='center',
                                       ha='left')
        text_entry_ax.set_axis_off()
        # append button at bottom
        button_ax = div.append_axes(position='bottom', size=Fixed(3 * pad),
                                    pad=Fixed(pad))
        fig.button = Button(button_ax, 'Add new label')
        fig.button.on_clicked(self._add_annotation_label)
        plt_show(fig=fig)
        # add "draggable" checkbox
        drag_ax_height = 3 * pad
        drag_ax = div.append_axes('bottom', size=Fixed(drag_ax_height),
                                  pad=Fixed(pad), aspect='equal')
        checkbox = CheckButtons(drag_ax, labels=('Draggable edges?',),
                                actives=(self.mne.draggable_annotations,))
        checkbox.on_clicked(self._toggle_draggable_annotations)
        fig.mne.drag_checkbox = checkbox
        # reposition & resize axes
        width_in, height_in = fig.get_size_inches()
        width_ax = fig._inch_to_rel(width_in - 2 * pad)
        aspect = width_ax / fig._inch_to_rel(drag_ax_height)
        drag_ax.set_xlim(0, aspect)
        drag_ax.set_axis_off()
        # reposition & resize checkbox & label
        rect = checkbox.rectangles[0]
        _pad, _size = (0.2, 0.6)
        rect.set_bounds(_pad, _pad, _size, _size)
        lines = checkbox.lines[0]
        for line, direction in zip(lines, (1, -1)):
            line.set_xdata((_pad, _pad + _size)[::direction])
            line.set_ydata((_pad, _pad + _size))
        text = checkbox.labels[0]
        text.set(position=(3 * _pad + _size, 0.45), va='center')
        for artist in lines + (rect, text):
            artist.set_transform(drag_ax.transData)
        # setup interactivity in plot window
        col = ('#ff0000' if len(fig.mne.radio_ax.buttons.circles) < 1 else
               fig.mne.radio_ax.buttons.circles[0].get_edgecolor())
        # TODO: we would like useblit=True here, but MPL #9660 prevents it
        selector = SpanSelector(self.mne.ax_main, self._select_annotation_span,
                                'horizontal', minspan=0.1, useblit=False,
                                rectprops=dict(alpha=0.5, facecolor=col))
        self.mne.ax_main.selector = selector
        self.mne._callback_ids['motion_notify_event'] = \
            self.canvas.mpl_connect('motion_notify_event', self._hover)

    def _toggle_draggable_annotations(self, event):
        """Enable/disable draggable annotation edges."""
        self.mne.draggable_annotations = not self.mne.draggable_annotations

    def _update_annotation_fig(self):
        """Draw or redraw the radio buttons and annotation labels."""
        from matplotlib.widgets import RadioButtons
        # define shorthand variables
        fig = self.mne.fig_annotation
        ax = fig.mne.radio_ax
        # get all the labels
        labels = list(set(self.mne.inst.annotations.description))
        labels = np.union1d(labels, self.mne.new_annotation_labels)
        # compute new figsize
        width, var_height, fixed_height, pad = \
            self._compute_annotation_figsize(len(labels))
        fig.set_size_inches(width, var_height + fixed_height, forward=True)
        # populate center axes with labels & radio buttons
        ax.clear()
        title = 'Existing labels:' if len(labels) else 'No existing labels'
        ax.set_title(title, size=None, loc='left')
        ax.buttons = RadioButtons(ax, labels)
        # adjust xlim to keep equal aspect & full width (keep circles round)
        aspect = (width - 2 * pad) / var_height
        ax.set_xlim((0, aspect))
        # style the buttons & adjust spacing
        radius = 0.15
        circles = ax.buttons.circles
        for circle, label in zip(circles, ax.buttons.labels):
            circle.set_transform(ax.transData)
            center = ax.transData.inverted().transform(
                ax.transAxes.transform((0.1, 0)))
            # XXX older MPL doesn't have circle.set_center
            circle.center = (center[0], circle.center[1])
            circle.set_edgecolor(
                self.mne.annotation_segment_colors[label.get_text()])
            circle.set_linewidth(4)
            circle.set_radius(radius / len(labels))
        # style the selected button
        if len(labels):
            fig._set_active_button(0)
        # add event listeners
        ax.buttons.disconnect_events()  # clear MPL default listeners
        ax.buttons.on_clicked(fig._radiopress)
        ax.buttons.connect_event('button_press_event', fig._click_override)

    def _toggle_annotation_fig(self):
        """Show/hide the annotation dialog window."""
        if self.mne.fig_annotation is None:
            self._create_annotation_fig()
        else:
            from matplotlib.pyplot import close
            close(self.mne.fig_annotation)

    def _compute_annotation_figsize(self, n_labels):
        """Adapt size of Annotation UI to accommodate the number of buttons.

        self._create_annotation_fig() implements the following:

        Fixed part of height:
        0.1  top margin
        1.0  instructions
        0.5  padding below instructions
        ---  (variable-height axis for label list)
        0.1  padding above text entry
        0.3  text entry
        0.1  padding above button
        0.3  button
        0.1  padding above checkbox
        0.3  checkbox
        0.1  bottom margin
        ------------------------------------------
        2.9  total fixed height
        """
        pad = 0.1
        width = 4.5
        var_height = max(pad, 0.7 * n_labels)
        fixed_height = 2.9
        return (width, var_height, fixed_height, pad)

    def _add_annotation_label(self, event):
        """Add new annotation description."""
        text = self.mne.fig_annotation.label.get_text()
        self.mne.new_annotation_labels.append(text)
        self._setup_annotation_colors()
        self._update_annotation_fig()
        # automatically activate new label's radio button
        idx = [label.get_text() for label in
               self.mne.fig_annotation.mne.radio_ax.buttons.labels].index(text)
        self.mne.fig_annotation._set_active_button(idx)
        # simulate a click on the radiobutton → update the span selector color
        self.mne.fig_annotation._radiopress(event=None)
        # reset the text entry box's text
        self.mne.fig_annotation.label.set_text('BAD_')

    def _setup_annotation_colors(self):
        """Set up colors for annotations."""
        raw = self.mne.inst
        segment_colors = getattr(self.mne, 'annotation_segment_colors', dict())
        # sort the segments by start time
        ann_order = raw.annotations.onset.argsort(axis=0)
        descriptions = raw.annotations.description[ann_order]
        color_keys = np.union1d(descriptions, self.mne.new_annotation_labels)
        colors, red = _get_color_list(annotations=True)
        color_cycle = cycle(colors)
        for key, color in segment_colors.items():
            if color != red and key in color_keys:
                next(color_cycle)
        for idx, key in enumerate(color_keys):
            if key in segment_colors:
                continue
            elif key.lower().startswith('bad') or \
                    key.lower().startswith('edge'):
                segment_colors[key] = red
            else:
                segment_colors[key] = next(color_cycle)
        self.mne.annotation_segment_colors = segment_colors

    def _select_annotation_span(self, vmin, vmax):
        """Handle annotation span selector."""
        onset = _sync_onset(self.mne.inst, vmin, True) - self.mne.first_time
        duration = vmax - vmin
        buttons = self.mne.fig_annotation.mne.radio_ax.buttons
        labels = [label.get_text() for label in buttons.labels]
        active_idx = labels.index(buttons.value_selected)
        _merge_annotations(onset, onset + duration, labels[active_idx],
                           self.mne.inst.annotations)
        self._draw_annotations()
        self.canvas.draw_idle()

    def _remove_annotation_hover_line(self):
        """Remove annotation line from the plot and reactivate selector."""
        if self.mne.annotation_hover_line is not None:
            self.mne.annotation_hover_line.remove()
            self.mne.annotation_hover_line = None
            self.mne.ax_main.selector.active = True
            self.canvas.draw()

    def _modify_annotation(self, old_x, new_x):
        """Modify annotation."""
        segment = np.array(np.where(self.mne.annotation_segments == old_x))
        if segment.shape[1] == 0:
            return
        raw = self.mne.inst
        annotations = raw.annotations
        first_time = self.mne.first_time
        idx = [segment[0][0], segment[1][0]]
        onset = _sync_onset(raw, self.mne.annotation_segments[idx[0]][0], True)
        ann_idx = np.where(annotations.onset == onset - first_time)[0]
        if idx[1] == 0:  # start of annotation
            onset = _sync_onset(raw, new_x, True) - first_time
            duration = annotations.duration[ann_idx] + old_x - new_x
        else:  # end of annotation
            onset = annotations.onset[ann_idx]
            duration = _sync_onset(raw, new_x, True) - onset - first_time
        if duration < 0:
            onset += duration
            duration *= -1.
        _merge_annotations(onset, onset + duration,
                           annotations.description[ann_idx],
                           annotations, ann_idx)
        self._draw_annotations()
        self._remove_annotation_hover_line()
        self.canvas.draw_idle()

    def _clear_annotations(self):
        """Clear all annotations from the figure."""
        for annot in self.mne.annotations[::-1]:
            self.mne.ax_main.collections.remove(annot)
            self.mne.annotations.remove(annot)
        for annot in self.mne.hscroll_annotations[::-1]:
            self.mne.ax_hscroll.collections.remove(annot)
            self.mne.hscroll_annotations.remove(annot)
        for text in self.mne.annotation_texts[::-1]:
            self.mne.ax_main.texts.remove(text)
            self.mne.annotation_texts.remove(text)

    def _draw_annotations(self):
        """Draw (or redraw) the annotation spans."""
        self._clear_annotations()
        self._update_annotation_segments()
        segments = self.mne.annotation_segments
        times = self.mne.times
        ax = self.mne.ax_main
        ylim = ax.get_ylim()
        for idx, (start, end) in enumerate(segments):
            descr = self.mne.inst.annotations.description[idx]
            segment_color = self.mne.annotation_segment_colors[descr]
            kwargs = dict(color=segment_color, alpha=0.3,
                          zorder=self.mne.zorder['ann'])
            # draw all segments on ax_hscroll
            annot = self.mne.ax_hscroll.fill_betweenx((0, 1), start, end,
                                                      **kwargs)
            self.mne.hscroll_annotations.append(annot)
            # draw only visible segments on ax_main
            visible_segment = np.clip([start, end], times[0], times[-1])
            if np.diff(visible_segment) > 0:
                annot = ax.fill_betweenx(ylim, *visible_segment, **kwargs)
                self.mne.annotations.append(annot)
                xy = (visible_segment.mean(), ylim[1])
                text = ax.annotate(descr, xy, xytext=(0, 9),
                                   textcoords='offset points', ha='center',
                                   va='baseline', color=segment_color)
                self.mne.annotation_texts.append(text)

    def _update_annotation_segments(self):
        """Update the array of annotation start/end times."""
        segments = list()
        raw = self.mne.inst
        if len(raw.annotations):
            for idx, annot in enumerate(raw.annotations):
                annot_start = _sync_onset(raw, annot['onset'])
                annot_end = annot_start + max(annot['duration'],
                                              1 / self.mne.info['sfreq'])
                segments.append((annot_start, annot_end))
        self.mne.annotation_segments = np.array(segments)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # CHANNEL SELECTION GUI
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _create_selection_fig(self):
        """Create channel selection dialog window."""
        from matplotlib.colors import to_rgb
        from matplotlib.widgets import RadioButtons
        from matplotlib.gridspec import GridSpec
        # make figure
        fig = self._new_child_figure(figsize=(3, 7),
                                     FigureClass=MNESelectionFigure,
                                     fig_name='fig_selection',
                                     window_title='Channel selection')
        # XXX when matplotlib 3.3 is min version, replace this with
        # XXX gs = fig.add_gridspec(15, 1)
        gs = GridSpec(nrows=15, ncols=1)
        # add sensor plot at top
        fig.mne.sensor_ax = fig.add_subplot(gs[:5])
        plot_sensors(self.mne.info, kind='select', ch_type='all', title='',
                     axes=fig.mne.sensor_ax, ch_groups=self.mne.group_by,
                     show=False)
        fig.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99)
        # style the sensors so the selection is easier to distinguish
        fig.lasso.linewidth_selected = 2
        self._update_highlighted_sensors()
        # add radio button axes
        radio_ax = fig.add_subplot(gs[5:-3], frame_on=False, aspect='equal')
        fig.mne.radio_ax = radio_ax
        selections_dict = self.mne.ch_selections
        selections_dict.update(Custom=np.array([], dtype=int))  # for lasso
        labels = list(selections_dict)
        # make & style the radio buttons
        activecolor = to_rgb(self.mne.fgcolor) + (0.5,)
        radio_ax.buttons = RadioButtons(radio_ax, labels,
                                        activecolor=activecolor)
        fig.mne.old_selection = 0
        for circle in radio_ax.buttons.circles:
            circle.set_radius(0.25 / len(labels))
            circle.set_linewidth(2)
            circle.set_edgecolor(self.mne.fgcolor)
        fig._style_radio_buttons_butterfly()
        # add instructions at bottom
        instructions = (
            'To use a custom selection, first click-drag on the sensor plot '
            'to "lasso" the sensors you want to select, or hold Ctrl while '
            'clicking individual sensors. Holding Ctrl while click-dragging '
            'allows a lasso selection adding to (rather than replacing) the '
            'existing selection.')
        instructions_ax = fig.add_subplot(gs[-3:], frame_on=False)
        instructions_ax.text(0.04, 0.08, instructions, va='bottom', ha='left',
                             ma='left', wrap=True)
        instructions_ax.set_axis_off()
        # add event listeners
        radio_ax.buttons.on_clicked(fig._radiopress)
        fig.canvas.mpl_connect('lasso_event', fig._set_custom_selection)

    def _change_selection_vscroll(self, event):
        """Handle clicks on vertical scrollbar when using selections."""
        buttons = self.mne.fig_selection.mne.radio_ax.buttons
        labels = [label.get_text() for label in buttons.labels]
        offset = 0
        selections_dict = self.mne.ch_selections
        for idx, label in enumerate(labels):
            offset += len(selections_dict[label])
            if event.ydata < offset:
                with _events_off(buttons):
                    buttons.set_active(idx)
                self.mne.fig_selection._radiopress(event)
                return

    def _update_selection(self):
        """Update visible channels based on selection dialog interaction."""
        selections_dict = self.mne.ch_selections
        fig = self.mne.fig_selection
        buttons = fig.mne.radio_ax.buttons
        label = buttons.value_selected
        labels = [_label.get_text() for _label in buttons.labels]
        self.mne.fig_selection.mne.old_selection = labels.index(label)
        self.mne.picks = selections_dict[label]
        self.mne.n_channels = len(self.mne.picks)
        self._update_highlighted_sensors()
        # if "Vertex" is defined, some channels appear twice, so if
        # "Vertex" is selected, ch_start should be the *first* match;
        # otherwise it should be the *last* match (since "Vertex" is
        # always the first selection group, if it exists).
        index = 0 if label == 'Vertex' else -1
        ch_order = np.concatenate(list(selections_dict.values()))
        ch_start = np.where(ch_order == self.mne.picks[0])[0][index]
        self.mne.ch_start = ch_start
        self._update_trace_offsets()
        self._update_vscroll()
        self._redraw(annotations=True)

    def _make_butterfly_selections_dict(self):
        """Make an altered copy of the selections dict for butterfly mode."""
        from ..utils import _get_stim_channel
        selections_dict = deepcopy(self.mne.ch_selections)
        # remove potential duplicates
        for selection_group in ('Vertex', 'Custom'):
            selections_dict.pop(selection_group, None)
        # if present, remove stim channel from non-misc selection groups
        stim_ch = _get_stim_channel(None, self.mne.info, raise_error=False)
        if len(stim_ch):
            stim_pick = self.mne.ch_names.tolist().index(stim_ch[0])
            for _sel, _picks in selections_dict.items():
                if _sel != 'Misc':
                    stim_mask = np.in1d(_picks, [stim_pick], invert=True)
                    selections_dict[_sel] = np.array(_picks)[stim_mask]
        return selections_dict

    def _update_highlighted_sensors(self):
        """Update the sensor plot to show what is selected."""
        inds = np.in1d(self.mne.fig_selection.lasso.ch_names,
                       self.mne.ch_names[self.mne.picks]).nonzero()[0]
        self.mne.fig_selection.lasso.select_many(inds)

    def _update_bad_sensors(self, pick, mark_bad):
        """Update the sensor plot to reflect (un)marked bad channels."""
        # replicate plotting order from plot_sensors(), to get index right
        sensor_picks = list()
        ch_indices = channel_indices_by_type(self.mne.info)
        for this_type in _DATA_CH_TYPES_SPLIT:
            if this_type in self.mne.ch_types:
                sensor_picks.extend(ch_indices[this_type])
        sensor_idx = np.in1d(sensor_picks, pick).nonzero()[0]
        # change the sensor color
        fig = self.mne.fig_selection
        fig.lasso.ec[sensor_idx, 0] = float(mark_bad)  # change R of RGBA array
        fig.lasso.collection.set_edgecolors(fig.lasso.ec)
        fig.canvas.draw_idle()

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # PROJECTORS & BAD CHANNELS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _create_proj_fig(self):
        """Create the projectors dialog window."""
        from matplotlib.widgets import Button, CheckButtons

        projs = self.mne.projs
        labels = [p['desc'] for p in projs]
        for ix, active in enumerate(self.mne.projs_active):
            if active:
                labels[ix] += ' (already applied)'
        # make figure
        width = max([4.5, max([len(label) for label in labels]) / 8 + 0.5])
        height = (len(projs) + 1) / 6 + 1.5
        fig = self._new_child_figure(figsize=(width, height),
                                     fig_name='fig_proj',
                                     window_title='SSP projection vectors')
        # make axes
        offset = (1 / 6 / height)
        position = (0, offset, 1, 0.8 - offset)
        ax = fig.add_axes(position, frame_on=False, aspect='equal')
        # make title
        first_line = ('Projectors already applied to the data are dimmed.\n'
                      if any(self.mne.projs_active) else '')
        second_line = 'Projectors marked with "X" are active on the plot.'
        ax.set_title(f'{first_line}{second_line}')
        # draw checkboxes
        checkboxes = CheckButtons(ax, labels=labels, actives=self.mne.projs_on)
        # gray-out already applied projectors
        for label, rect, lines in zip(checkboxes.labels,
                                      checkboxes.rectangles,
                                      checkboxes.lines):
            if label.get_text().endswith('(already applied)'):
                label.set_color('0.5')
                rect.set_edgecolor('0.7')
                [x.set_color('0.7') for x in lines]
            rect.set_linewidth(1)
        # add "toggle all" button
        ax_all = fig.add_axes((0.25, 0.01, 0.5, offset), frame_on=True)
        fig.mne.proj_all = Button(ax_all, 'Toggle all')
        # add event listeners
        checkboxes.on_clicked(self._toggle_proj_checkbox)
        fig.mne.proj_all.on_clicked(partial(self._toggle_proj_checkbox,
                                            toggle_all=True))
        # save params
        fig.mne.proj_checkboxes = checkboxes
        # show figure
        self.mne.fig_proj.canvas.draw()
        plt_show(fig=self.mne.fig_proj, warn=False)

    def _toggle_proj_fig(self, event=None):
        """Show/hide the projectors dialog window."""
        if self.mne.fig_proj is None:
            self._create_proj_fig()
        else:
            from matplotlib.pyplot import close
            close(self.mne.fig_proj)

    def _toggle_proj_checkbox(self, event, toggle_all=False):
        """Perform operations when proj boxes clicked."""
        on = self.mne.projs_on
        applied = self.mne.projs_active
        fig = self.mne.fig_proj
        new_state = (np.full_like(on, not all(on)) if toggle_all else
                     np.array(fig.mne.proj_checkboxes.get_status()))
        # update Xs when toggling all
        if toggle_all:
            with _events_off(fig.mne.proj_checkboxes):
                for ix in np.where(on != new_state)[0]:
                    fig.mne.proj_checkboxes.set_active(ix)
        # don't allow disabling already-applied projs
        with _events_off(fig.mne.proj_checkboxes):
            for ix in np.where(applied)[0]:
                if not new_state[ix]:
                    fig.mne.proj_checkboxes.set_active(ix)
            new_state[applied] = True
        # update the data if necessary
        if not np.array_equal(on, new_state):
            self.mne.projs_on = new_state
            self._update_projector()
            self._redraw()

    def _update_projector(self):
        """Update the data after projectors (or bads) have changed."""
        inds = np.where(self.mne.projs_on)[0]  # doesn't include "active" projs
        # copy projs from full list (self.mne.projs) to info object
        self.mne.info['projs'] = [deepcopy(self.mne.projs[ix]) for ix in inds]
        # compute the projection operator
        proj, wh_chs = _setup_plot_projector(self.mne.info, self.mne.noise_cov,
                                             True, self.mne.use_noise_cov)
        self.mne.whitened_ch_names = list(wh_chs)
        self.mne.projector = proj

    def _toggle_bad_channel(self, idx):
        """Mark/unmark bad channels; `idx` is index of *visible* channels."""
        pick = self.mne.picks[idx]
        ch_name = self.mne.ch_names[pick]
        # add/remove from bads list
        bads = self.mne.info['bads']
        marked_bad = ch_name not in bads
        if marked_bad:
            bads.append(ch_name)
            color = self.mne.ch_color_bad
        else:
            while ch_name in bads:  # to make sure duplicates are removed
                bads.remove(ch_name)
            color = self.mne.ch_colors[idx]
        self.mne.info['bads'] = bads
        # update sensor color (if in selection mode)
        if self.mne.fig_selection is not None:
            self._update_bad_sensors(pick, marked_bad)
        # update vscroll color
        vscroll_idx = (self.mne.ch_order == pick).nonzero()[0]
        for _idx in vscroll_idx:
            self.mne.ax_vscroll.patches[_idx].set_color(color)
        # redraw
        self._update_projector()
        self._redraw()

    def _toggle_bad_epoch(self, event):
        """Mark/unmark bad epochs."""
        epoch_num = self._get_epoch_num_from_time(event.xdata)
        epoch_ix = self.mne.inst.selection.tolist().index(epoch_num)
        if epoch_num in self.mne.bad_epochs:
            self.mne.bad_epochs.remove(epoch_num)
            color = 'none'
        else:
            self.mne.bad_epochs.append(epoch_num)
            self.mne.bad_epochs.sort()
            color = self.mne.epoch_color_bad
        self.mne.ax_hscroll.patches[epoch_ix].set_color(color)
        self._redraw(update_data=False)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # SCROLLBARS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _toggle_scrollbars(self):
        """Show or hide scrollbars (A.K.A. zen mode)."""
        # grow/shrink main axes to take up space from (or make room for)
        # scrollbars. We can't use ax.set_position() because axes are
        # locatable, so we use subplots_adjust
        should_show = not self.mne.scrollbars_visible
        margins = {side: getattr(self.subplotpars, side)
                   for side in ('left', 'bottom', 'right', 'top')}
        # if should_show, bottom margin moves up; right margin moves left
        margins['bottom'] += (1 if should_show else -1) * self.mne.zen_h
        margins['right'] += (-1 if should_show else 1) * self.mne.zen_w
        # squeeze a bit more because we don't need space for xlabel now
        self.subplots_adjust(**margins)
        # handle x-axis label
        self.mne.zen_xlabel.set_visible(not should_show)
        # show/hide other UI elements
        for elem in ('ax_hscroll', 'ax_vscroll', 'ax_proj', 'ax_help'):
            if elem == 'ax_vscroll' and self.mne.butterfly:
                continue
            # sometimes we don't have a proj button (ax_proj)
            if getattr(self.mne, elem, None) is not None:
                getattr(self.mne, elem).set_visible(should_show)
        self.mne.scrollbars_visible = should_show

    def _update_vscroll(self):
        """Update the vertical scrollbar (channel) selection indicator."""
        self.mne.vsel_patch.set_xy((0, self.mne.ch_start))
        self.mne.vsel_patch.set_height(self.mne.n_channels)
        self._update_yaxis_labels()

    def _update_hscroll(self):
        """Update the horizontal scrollbar (time) selection indicator."""
        self.mne.hsel_patch.set_xy((self.mne.t_start, 0))
        self.mne.hsel_patch.set_width(self.mne.duration)

    def _check_update_hscroll_clicked(self, event):
        """Handle clicks on horizontal scrollbar."""
        time = event.xdata - self.mne.duration / 2
        max_time = (self.mne.n_times / self.mne.info['sfreq'] +
                    self.mne.first_time - self.mne.duration)
        time = np.clip(time, self.mne.first_time, max_time)
        if self.mne.is_epochs:
            ix = np.searchsorted(self.mne.boundary_times[1:], time)
            time = self.mne.boundary_times[ix]
        if self.mne.t_start != time:
            self.mne.t_start = time
            self._update_hscroll()
            return True
        return False

    def _check_update_vscroll_clicked(self, event):
        """Update vscroll patch on click, return True if location changed."""
        new_ch_start = np.clip(
            int(round(event.ydata - self.mne.n_channels / 2)),
            0, len(self.mne.ch_order) - self.mne.n_channels)
        if self.mne.ch_start != new_ch_start:
            self.mne.ch_start = new_ch_start
            self._update_picks()
            self._update_vscroll()
            return True
        return False

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # SCALEBARS & Y-AXIS LABELS
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _show_scalebars(self):
        """Add channel scale bars."""
        for offset, pick in zip(self.mne.trace_offsets, self.mne.picks):
            this_name = self.mne.ch_names[pick]
            this_type = self.mne.ch_types[pick]
            if (this_type not in self.mne.scalebars and
                    this_type != 'stim' and
                    this_type in self.mne.scalings and
                    this_type in getattr(self.mne, 'units', {}) and
                    this_type in getattr(self.mne, 'unit_scalings', {}) and
                    this_name not in self.mne.info['bads'] and
                    this_name not in self.mne.whitened_ch_names):
                x = (self.mne.times[0] + self.mne.first_time,) * 2
                denom = 4 if self.mne.butterfly else 2
                y = tuple(np.array([-1, 1]) / denom + offset)
                self._draw_one_scalebar(x, y, this_type)

    def _hide_scalebars(self):
        """Remove channel scale bars."""
        for bar in self.mne.scalebars.values():
            self.mne.ax_main.lines.remove(bar)
        for text in self.mne.scalebar_texts.values():
            self.mne.ax_main.texts.remove(text)
        self.mne.scalebars = dict()
        self.mne.scalebar_texts = dict()

    def _toggle_scalebars(self, event):
        """Show/hide the scalebars."""
        if self.mne.scalebars_visible:
            self._hide_scalebars()
        else:
            self._update_picks()
            self._show_scalebars()
        # toggle
        self.mne.scalebars_visible = not self.mne.scalebars_visible
        self.canvas.draw_idle()

    def _draw_one_scalebar(self, x, y, ch_type):
        """Draw a scalebar."""
        from .utils import _simplify_float
        color = '#AA3377'  # purple
        kwargs = dict(color=color, zorder=self.mne.zorder['scalebar'])
        scaler = 1 if self.mne.butterfly else 2
        inv_norm = (scaler *
                    self.mne.scalings[ch_type] *
                    self.mne.unit_scalings[ch_type] /
                    self.mne.scale_factor)
        bar = self.mne.ax_main.plot(x, y, lw=4, **kwargs)[0]
        label = f'{_simplify_float(inv_norm)} {self.mne.units[ch_type]} '
        text = self.mne.ax_main.text(x[1], y[1], label, va='baseline',
                                     ha='right', size='xx-small', **kwargs)
        self.mne.scalebars[ch_type] = bar
        self.mne.scalebar_texts[ch_type] = text

    def _update_yaxis_labels(self):
        """Change the y-axis labels."""
        if self.mne.butterfly and self.mne.fig_selection is not None:
            exclude = ('Vertex', 'Custom')
            ticklabels = list(self.mne.ch_selections)
            keep_mask = np.in1d(ticklabels, exclude, invert=True)
            ticklabels = [t.replace('Left-', 'L-').replace('Right-', 'R-')
                          for t in ticklabels]  # avoid having to rotate labels
            ticklabels = np.array(ticklabels)[keep_mask]
        elif self.mne.butterfly:
            _, ixs, _ = np.intersect1d(_DATA_CH_TYPES_ORDER_DEFAULT,
                                       self.mne.ch_types, return_indices=True)
            ixs.sort()
            ticklabels = np.array(_DATA_CH_TYPES_ORDER_DEFAULT)[ixs]
        else:
            ticklabels = self.mne.ch_names[self.mne.picks]
        texts = self.mne.ax_main.set_yticklabels(ticklabels, picker=True)
        for text in texts:
            sty = ('italic' if text.get_text() in self.mne.whitened_ch_names
                   else 'normal')
            text.set_style(sty)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # DATA TRACES
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _toggle_butterfly(self):
        """Enter or leave butterfly mode."""
        self.mne.ax_vscroll.set_visible(self.mne.butterfly)
        self.mne.butterfly = not self.mne.butterfly
        self.mne.scale_factor *= 0.5 if self.mne.butterfly else 2.
        self._update_picks()
        self._update_trace_offsets()
        self._redraw(annotations=True)
        if self.mne.vline_visible:
            self._blit_vline(True)
        if self.mne.fig_selection is not None:
            self.mne.fig_selection._style_radio_buttons_butterfly()

    def _update_picks(self):
        """Compute which channel indices to show."""
        if self.mne.butterfly and self.mne.ch_selections is not None:
            selections_dict = self._make_butterfly_selections_dict()
            self.mne.picks = np.concatenate(tuple(selections_dict.values()))
        elif self.mne.butterfly:
            self.mne.picks = np.arange(self.mne.ch_names.shape[0])
        else:
            _slice = slice(self.mne.ch_start,
                           self.mne.ch_start + self.mne.n_channels)
            self.mne.picks = self.mne.ch_order[_slice]
            self.mne.n_channels = len(self.mne.picks)

    def _get_epoch_num_from_time(self, time):
        epoch_nums = self.mne.inst.selection
        return epoch_nums[np.searchsorted(self.mne.boundary_times[1:], time)]

    def _load_data(self, start=None, stop=None):
        """Retrieve the bit of data we need for plotting."""
        if 'raw' in (self.mne.instance_type, self.mne.ica_type):
            return self.mne.inst[:, start:stop]
        else:
            ix = np.searchsorted(self.mne.boundary_times, self.mne.t_start)
            item = slice(ix, ix + self.mne.n_epochs)
            data = np.concatenate(self.mne.inst.get_data(item=item), axis=-1)
            times = np.arange(len(self.mne.inst) * len(self.mne.inst.times)
                              )[start:stop] / self.mne.info['sfreq']
            return data, times

    def _update_data(self):
        """Update self.mne.data after user interaction."""
        from ..filter import _overlap_add_filter, _filtfilt
        # update time
        start_sec = self.mne.t_start - self.mne.first_time
        stop_sec = start_sec + self.mne.duration
        if self.mne.is_epochs:
            start, stop = np.round(np.array([start_sec, stop_sec])
                                   * self.mne.info['sfreq']).astype(int)
        else:
            start, stop = self.mne.inst.time_as_index((start_sec, stop_sec))
        # get the data
        data, times = self._load_data(start, stop)
        # apply projectors
        if self.mne.projector is not None:
            data = self.mne.projector @ data
        # get only the channels we're displaying
        picks = self.mne.picks
        data = data[picks]
        # remove DC
        if self.mne.remove_dc:
            data -= data.mean(axis=1, keepdims=True)
        # filter (with same defaults as raw.filter())
        if self.mne.filter_coefs is not None:
            starts, stops = self.mne.filter_bounds
            mask = (starts < stop) & (stops > start)
            starts = np.maximum(starts[mask], start) - start
            stops = np.minimum(stops[mask], stop) - start
            for _start, _stop in zip(starts, stops):
                _picks = np.where(np.in1d(picks, self.mne.picks_data))
                this_data = data[_picks, _start:_stop]
                if isinstance(self.mne.filter_coefs, np.ndarray):  # FIR
                    this_data = _overlap_add_filter(
                        this_data, self.mne.filter_coefs, copy=False)
                else:  # IIR
                    this_data = _filtfilt(
                        this_data, self.mne.filter_coefs, None, 1, False)
                data[_picks, _start:_stop] = this_data
        # scale the data for display in a 1-vertical-axis-unit slot
        this_names = self.mne.ch_names[picks]
        this_types = self.mne.ch_types[picks]
        stims = this_types == 'stim'
        white = np.logical_and(np.in1d(this_names, self.mne.whitened_ch_names),
                               np.in1d(this_names, self.mne.info['bads'],
                                       invert=True))
        norms = np.vectorize(self.mne.scalings.__getitem__)(this_types)
        norms[stims] = data[stims].max(axis=-1)
        norms[white] = self.mne.scalings['whitened']
        norms[norms == 0] = 1
        data /= 2 * norms[:, np.newaxis]
        self.mne.data = data
        self.mne.times = times

    def _update_trace_offsets(self):
        """Compute viewport height and adjust offsets."""
        # simultaneous selection and butterfly modes
        if self.mne.butterfly and self.mne.ch_selections is not None:
            self._update_picks()
            selections_dict = self._make_butterfly_selections_dict()
            n_offsets = len(selections_dict)
            sel_order = list(selections_dict)
            offsets = np.array([])
            for pick in self.mne.picks:
                for sel in sel_order:
                    if pick in selections_dict[sel]:
                        offsets = np.append(offsets, sel_order.index(sel))
        # butterfly only
        elif self.mne.butterfly:
            unique_ch_types = set(self.mne.ch_types)
            n_offsets = len(unique_ch_types)
            ch_type_order = [_type for _type in _DATA_CH_TYPES_ORDER_DEFAULT
                             if _type in unique_ch_types]
            offsets = np.array([ch_type_order.index(ch_type)
                                for ch_type in self.mne.ch_types])
        # normal mode
        else:
            n_offsets = self.mne.n_channels
            offsets = np.arange(n_offsets, dtype=float)
        # update ylim, ticks, vertline, and scrollbar patch
        ylim = (n_offsets - 0.5, -0.5)  # inverted y axis → new chs at bottom
        self.mne.ax_main.set_ylim(ylim)
        self.mne.ax_main.set_yticks(np.unique(offsets))
        self.mne.vsel_patch.set_height(self.mne.n_channels)
        # store new offsets, update axis labels
        self.mne.trace_offsets = offsets
        self._update_yaxis_labels()

    def _draw_traces(self):
        """Draw (or redraw) the channel data."""
        from matplotlib.colors import to_rgba_array
        from matplotlib.patches import Rectangle
        # clear scalebars
        if self.mne.scalebars_visible:
            self._hide_scalebars()
        # get info about currently visible channels
        picks = self.mne.picks
        ch_names = self.mne.ch_names[picks]
        ch_types = self.mne.ch_types[picks]
        bad_bool = np.in1d(ch_names, self.mne.info['bads'])
        # colors
        good_ch_colors = [self.mne.ch_color_dict[_type] for _type in ch_types]
        ch_colors = to_rgba_array(
            [self.mne.ch_color_bad if _bad else _color
             for _bad, _color in zip(bad_bool, good_ch_colors)])
        self.mne.ch_colors = np.array(good_ch_colors)  # use for unmarking bads
        labels = self.mne.ax_main.yaxis.get_ticklabels()
        if self.mne.butterfly:
            for label in labels:
                label.set_color(self.mne.fgcolor)
        else:
            for label, color in zip(labels, ch_colors):
                label.set_color(color)
        # decim
        decim = np.ones_like(picks)
        data_picks_mask = np.in1d(picks, self.mne.picks_data)
        decim[data_picks_mask] = self.mne.decim
        # decim can vary by channel type, so compute different `times` vectors
        decim_times = {decim_value:
                       self.mne.times[::decim_value] + self.mne.first_time
                       for decim_value in set(decim)}
        # add more traces if needed
        n_picks = len(picks)
        if n_picks > len(self.mne.traces):
            n_new_chs = n_picks - len(self.mne.traces)
            new_traces = self.mne.ax_main.plot(np.full((1, n_new_chs), np.nan),
                                               **self.mne.trace_kwargs)
            self.mne.traces.extend(new_traces)
        # remove extra traces if needed
        extra_traces = self.mne.traces[n_picks:]
        for trace in extra_traces:
            self.mne.ax_main.lines.remove(trace)
        self.mne.traces = self.mne.traces[:n_picks]

        # check for bad epochs
        time_range = (self.mne.times + self.mne.first_time)[[0, -1]]
        if self.mne.instance_type == 'epochs':
            epoch_ix = np.searchsorted(self.mne.boundary_times, time_range)
            epoch_ix = np.arange(epoch_ix[0], epoch_ix[1])
            epoch_nums = self.mne.inst.selection[epoch_ix[0]:epoch_ix[-1] + 1]
            visible_bad_epochs = epoch_nums[
                np.in1d(epoch_nums, self.mne.bad_epochs).nonzero()]
            while len(self.mne.epoch_traces):
                _trace = self.mne.epoch_traces.pop(-1)
                self.mne.ax_main.lines.remove(_trace)
            # handle custom epoch colors (for autoreject integration)
            if self.mne.epoch_colors is None:
                # shape: n_traces × RGBA → n_traces × n_epochs × RGBA
                custom_colors = np.tile(ch_colors[:, None, :],
                                        (1, self.mne.n_epochs, 1))
            else:
                custom_colors = np.empty((len(self.mne.picks),
                                          self.mne.n_epochs, 4))
                for ii, _epoch_ix in enumerate(epoch_ix):
                    this_colors = self.mne.epoch_colors[_epoch_ix]
                    custom_colors[:, ii] = to_rgba_array([this_colors[_ch]
                                                          for _ch in picks])
            # override custom color on bad epochs
            for _bad in visible_bad_epochs:
                _ix = epoch_nums.tolist().index(_bad)
                _cols = np.array([self.mne.epoch_color_bad,
                                  self.mne.ch_color_bad])[bad_bool.astype(int)]
                custom_colors[:, _ix] = to_rgba_array(_cols)

        # update traces
        ylim = self.mne.ax_main.get_ylim()
        for ii, line in enumerate(self.mne.traces):
            this_name = ch_names[ii]
            this_type = ch_types[ii]
            this_offset = self.mne.trace_offsets[ii]
            this_times = decim_times[decim[ii]]
            this_data = this_offset - self.mne.data[ii] * self.mne.scale_factor
            this_data = this_data[..., ::decim[ii]]
            # clip
            if self.mne.clipping == 'clamp':
                this_data = np.clip(this_data, -0.5, 0.5)
            elif self.mne.clipping is not None:
                clip = self.mne.clipping * (0.2 if self.mne.butterfly else 1)
                bottom = max(this_offset - clip, ylim[1])
                height = min(2 * clip, ylim[0] - bottom)
                rect = Rectangle(xy=np.array([time_range[0], bottom]),
                                 width=time_range[1] - time_range[0],
                                 height=height,
                                 transform=self.mne.ax_main.transData)
                line.set_clip_path(rect)
            # prep z order
            is_bad_ch = this_name in self.mne.info['bads']
            this_z = self.mne.zorder['bads' if is_bad_ch else 'data']
            if self.mne.butterfly and not is_bad_ch:
                this_z = self.mne.zorder.get(this_type, this_z)
            # plot each trace multiple times to get the desired epoch coloring.
            # use masked arrays to plot discontinuous epochs that have the same
            # color in a single plot() call.
            if self.mne.instance_type == 'epochs':
                this_colors = custom_colors[ii]
                for cix, color in enumerate(np.unique(this_colors, axis=0)):
                    bool_ixs = (this_colors == color).all(axis=1)
                    mask = np.zeros_like(this_times, dtype=bool)
                    _starts = self.mne.boundary_times[epoch_ix][bool_ixs]
                    _stops = self.mne.boundary_times[epoch_ix + 1][bool_ixs]
                    for _start, _stop in zip(_starts, _stops):
                        _mask = np.logical_and(_start < this_times,
                                               this_times <= _stop)
                        mask = mask | _mask
                    _times = np.ma.masked_array(this_times, mask=~mask)
                    # always use the existing traces first
                    if cix == 0:
                        line.set_xdata(_times)
                        line.set_ydata(this_data)
                        line.set_color(color)
                        line.set_zorder(this_z)
                    else:  # make new traces as needed
                        _trace = self.mne.ax_main.plot(
                            _times, this_data, color=color, zorder=this_z,
                            **self.mne.trace_kwargs)
                        self.mne.epoch_traces.extend(_trace)
            else:
                line.set_xdata(this_times)
                line.set_ydata(this_data)
                line.set_color(ch_colors[ii])
                line.set_zorder(this_z)
        # update xlim
        self.mne.ax_main.set_xlim(*time_range)
        # draw scalebars maybe
        if self.mne.scalebars_visible:
            self._show_scalebars()
        # redraw event lines
        if self.mne.event_times is not None:
            self._draw_event_lines()

    def _redraw(self, update_data=True, annotations=False):
        """Redraw (convenience method for frequently grouped actions)."""
        if update_data:
            self._update_data()
        self._draw_traces()
        if annotations and not self.mne.is_epochs:
            self._draw_annotations()
        self.canvas.draw_idle()
        self.canvas.flush_events()
        self.mne.bg = self.canvas.copy_from_bbox(self.bbox)
        if self.mne.vline_visible:
            self._blit_vline(True)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # EVENT LINES AND MARKER LINES
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def _draw_event_lines(self):
        """Draw the event lines and their labels."""
        from matplotlib.colors import to_rgba_array
        from matplotlib.collections import LineCollection
        if self.mne.event_nums is not None:
            mask = np.logical_and(self.mne.event_times >= self.mne.times[0],
                                  self.mne.event_times <= self.mne.times[-1])
            this_event_times = self.mne.event_times[mask]
            this_event_nums = self.mne.event_nums[mask]
            n_visible_events = len(this_event_times)
            colors = to_rgba_array([self.mne.event_color_dict[n]
                                    for n in this_event_nums])
            # create event lines
            ylim = self.mne.ax_main.get_ylim()
            xs = np.repeat(this_event_times, 2)
            ys = np.tile(ylim, n_visible_events)
            segs = np.vstack([xs, ys]).T.reshape(n_visible_events, 2, 2)
            event_lines = LineCollection(segs, linewidths=0.5, colors=colors,
                                         zorder=self.mne.zorder['events'])
            self.mne.ax_main.add_collection(event_lines)
            self.mne.event_lines = event_lines
            # create event labels
            while len(self.mne.event_texts):
                text = self.mne.event_texts.pop()
                self.mne.ax_main.texts.remove(text)
            for _t, _n, _c in zip(this_event_times, this_event_nums, colors):
                label = self.mne.event_id_rev.get(_n, _n)
                this_text = self.mne.ax_main.annotate(
                    label, (_t, ylim[1]), ha='center', va='baseline',
                    color=self.mne.fgcolor, xytext=(0, 2),
                    textcoords='offset points', fontsize=8)
                self.mne.event_texts.append(this_text)

    def _show_vline(self, xdata):
        """Show the vertical line."""
        if self.mne.is_epochs:
            # special case: changed view duration w/ "home" or "end" key
            # (no click event, hence no xdata)
            if xdata is None:
                xdata = np.array(self.mne.vline.get_segments())[0, 0, 0]
            # compute the (continuous) times for the lines on each epoch
            epoch_dur = np.diff(self.mne.boundary_times[:2])[0]
            rel_time = xdata % epoch_dur
            abs_time = self.mne.times[0]
            xs = np.arange(self.mne.n_epochs) * epoch_dur + abs_time + rel_time
            segs = np.array(self.mne.vline.get_segments())
            # handle changed view duration (n_segments != n_epochs)
            if segs.shape[0] != len(xs):
                segs = np.tile([[0.], [1.]], (len(xs), 1, 2))  # y values
            segs[..., 0] = np.tile(xs[:, None], 2)
            self.mne.vline.set_segments(segs)
            xdata = rel_time + self.mne.inst.times[0]  # for the text
        else:
            self.mne.vline.set_xdata(xdata)
            self.mne.vline_hscroll.set_xdata(xdata)
        self.mne.vline_text.set_text(f'{xdata:0.2f}  ')
        self._blit_vline(True)

    def _blit_vline(self, visible):
        """Restore or hide the vline after data change."""
        self.canvas.restore_region(self.mne.bg)
        for artist in (self.mne.vline, self.mne.vline_hscroll,
                       self.mne.vline_text):
            if artist is not None:
                artist.set_visible(visible)
                self.draw_artist(artist)
        self.canvas.blit()
        self.canvas.flush_events()
        self.mne.vline_visible = visible


class MNEPSDFigure(MNEFigure):
    """Interactive figure for power spectral density plots."""

    def __init__(self, inst, n_axes, figsize, **kwargs):
        super().__init__(figsize=figsize, inst=inst, **kwargs)

        # AXES: default margins (inches)
        l_margin = 0.8
        r_margin = 0.2
        b_margin = 0.65
        t_margin = 0.35
        # AXES: default margins (figure-relative coordinates)
        left = self._inch_to_rel(l_margin)
        right = 1 - self._inch_to_rel(r_margin)
        bottom = self._inch_to_rel(b_margin, horiz=False)
        top = 1 - self._inch_to_rel(t_margin, horiz=False)
        # AXES: make subplots
        axes = [self.add_subplot(n_axes, 1, 1)]
        for ix in range(1, n_axes):
            axes.append(self.add_subplot(n_axes, 1, ix + 1, sharex=axes[0]))
        self.subplotpars.update(left=left, bottom=bottom, top=top, right=right,
                                hspace=0.4)
        # save useful things
        self.mne.ax_list = axes

    def _resize(self, event):
        """Handle resize event."""
        old_width, old_height = self.mne.fig_size_px
        new_width, new_height = self._get_size_px()
        new_margins = _calc_new_margins(
            self, old_width, old_height, new_width, new_height)
        self.subplots_adjust(**new_margins)
        self.mne.fig_size_px = (new_width, new_height)


def _figure(toolbar=True, FigureClass=MNEFigure, **kwargs):
    """Instantiate a new figure."""
    from matplotlib import rc_context
    from matplotlib.pyplot import figure
    title = kwargs.pop('window_title', None)  # extract title before init
    rc = dict() if toolbar else dict(toolbar='none')
    with rc_context(rc=rc):
        fig = figure(FigureClass=FigureClass, **kwargs)
    if title is not None:
        _set_window_title(fig, title)
    return fig


def _browse_figure(inst, **kwargs):
    """Instantiate a new MNE browse-style figure."""
    from .utils import _get_figsize_from_config
    figsize = kwargs.pop('figsize', _get_figsize_from_config())
    fig = _figure(inst=inst, toolbar=False, FigureClass=MNEBrowseFigure,
                  figsize=figsize, **kwargs)
    # initialize zen mode (can't do in __init__ due to get_position() calls)
    fig.canvas.draw()
    fig.mne.fig_size_px = fig._get_size_px()
    fig.mne.zen_w = (fig.mne.ax_vscroll.get_position().xmax -
                     fig.mne.ax_main.get_position().xmax)
    fig.mne.zen_h = (fig.mne.ax_main.get_position().ymin -
                     fig.mne.ax_hscroll.get_position().ymin)
    # if scrollbars are supposed to start hidden, set to True and then toggle
    if not fig.mne.scrollbars_visible:
        fig.mne.scrollbars_visible = True
        fig._toggle_scrollbars()
    # add event callbacks
    fig._add_default_callbacks()
    return fig


def _psd_figure(inst, proj, picks, axes, area_mode, tmin, tmax, fmin, fmax,
                n_jobs, color, area_alpha, dB, estimate, average,
                spatial_colors, xscale, line_alpha, sphere, **kwargs):
    """Instantiate a new power spectral density figure."""
    from matplotlib.axes import Axes
    from .. import BaseEpochs
    from ..io import BaseRaw
    # triage kwargs for different PSD methods (raw→welch, epochs→multitaper)
    welch_kwargs = ('n_fft', 'n_overlap', 'reject_by_annotation')
    multitaper_kwargs = ('bandwidth', 'adaptive', 'low_bias', 'normalization')
    psd_kwargs = dict()
    for kw in welch_kwargs + multitaper_kwargs:
        if kw in kwargs:
            psd_kwargs[kw] = kwargs.pop(kw)
    if isinstance(inst, BaseRaw):
        psd_func = psd_welch
    elif isinstance(inst, BaseEpochs):
        psd_func = psd_multitaper
    else:
        raise TypeError('Expected an instance of Raw or Epochs, got '
                        f'{type(inst)}.')
    # arg checking
    if np.isfinite(fmax) and (fmax > inst.info['sfreq'] / 2):
        raise ValueError(
            f'Requested fmax ({fmax} Hz) must not exceed ½ the sampling '
            f'frequency of the data ({0.5 * inst.info["sfreq"]}).')
    _check_option('area_mode', area_mode, [None, 'std', 'range'])
    _check_option('xscale', xscale, ('log', 'linear'))
    sphere = _check_sphere(sphere, inst.info)
    picks = _picks_to_idx(inst.info, picks)
    titles = _handle_default('titles', None)
    units = _handle_default('units', None)
    scalings = _handle_default('scalings', None)
    # containers
    picks_list = list()
    units_list = list()
    titles_list = list()
    scalings_list = list()
    psd_list = list()
    # handle picks
    _user_picked = picks is not None
    allowed_ch_types = (_VALID_CHANNEL_TYPES if _user_picked else
                        _DATA_CH_TYPES_SPLIT)
    for ch_type in allowed_ch_types:
        pick_kwargs = dict(meg=False, ref_meg=False, exclude=[])
        if ch_type in ('mag', 'grad'):
            pick_kwargs['meg'] = ch_type
        elif ch_type in _FNIRS_CH_TYPES_SPLIT:
            pick_kwargs['fnirs'] = ch_type
        else:
            pick_kwargs[ch_type] = True
        these_picks = pick_types(inst.info, **pick_kwargs)
        these_picks = np.intersect1d(these_picks, picks)
        if len(these_picks) > 0:
            picks_list.append(these_picks)
            titles_list.append(titles[ch_type])
            units_list.append(units[ch_type])
            scalings_list.append(scalings[ch_type])
    del picks
    n_types = len(picks_list)
    if n_types == 0:
        raise RuntimeError('No data channels found')
    # handle user-provided axes
    if axes is not None:
        if isinstance(axes, Axes):
            axes = [axes]
        _validate_if_list_of_axes(axes, n_types)
        fig = axes[0].get_figure()
    else:
        figsize = kwargs.pop('figsize', (10, 2.5 * n_types + 1))
        fig = _figure(inst=inst, toolbar=False, FigureClass=MNEPSDFigure,
                      figsize=figsize, n_axes=n_types, **kwargs)
        fig.mne.fig_size_px = fig._get_size_px()  # can't do in __init__
        axes = fig.mne.ax_list
        # add event callbacks
        fig._add_default_callbacks()
    # don't add ylabels & titles if figure has unexpected number of axes
    make_label = len(axes) == len(fig.axes)
    # Plot Frequency [Hz] xlabel on the last axis
    xlabels_list = [False] * (n_types - 1) + [True]
    # compute PSDs
    for picks in picks_list:
        psd, freqs = psd_func(inst, tmin=tmin, tmax=tmax, picks=picks,
                              fmin=fmin, fmax=fmax, proj=proj, n_jobs=n_jobs,
                              **psd_kwargs)
        if isinstance(inst, BaseEpochs):
            psd = np.mean(psd, axis=0)
        psd_list.append(psd)
    # plot
    _plot_psd(inst, fig, freqs, psd_list, picks_list, titles_list, units_list,
              scalings_list, axes, make_label, color, area_mode, area_alpha,
              dB, estimate, average, spatial_colors, xscale, line_alpha,
              sphere, xlabels_list)
    return fig


def _calc_new_margins(fig, old_width, old_height, new_width, new_height):
    """Compute new figure-relative values to maintain fixed-size margins."""
    new_margins = dict()
    for side in ('left', 'right', 'bottom', 'top'):
        ratio = ((old_width / new_width) if side in ('left', 'right') else
                 (old_height / new_height))
        rel_dim = getattr(fig.subplotpars, side)
        if side in ('right', 'top'):
            new_margins[side] = 1 - ratio * (1 - rel_dim)
        else:
            new_margins[side] = ratio * rel_dim
    # gh-8304: don't allow resizing too small
    if (new_margins['bottom'] < new_margins['top'] and
            new_margins['left'] < new_margins['right']):
        return(new_margins)


@contextmanager
def _patched_canvas(fig):
    old_canvas = fig.canvas
    if fig.canvas is None:  # XXX old MPL (at least 3.0.3) does this for Agg
        fig.canvas = Bunch(mpl_connect=lambda event, callback: None)
    try:
        yield
    finally:
        fig.canvas = old_canvas
