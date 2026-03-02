import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly
import os
from tqdm import tqdm
import colorsys
import math
import re

from ._utils import prepare_wide_data


class _BarChartRace:
    
    def __init__(self, data_filename, data_is_wide, out_filename, orientation, sort, n_bars, fixed_order, fixed_max,
                 steps_per_period, period_length, end_period_pause, interpolate_period, 
                 period_label, period_template, period_summary_func, perpendicular_bar_func, 
                 title, bar_size, bar_textposition, bar_texttemplate, bar_label_font, 
                 tick_label_font, hovertemplate, slider, scale, bar_kwargs, layout_kwargs, 
                 write_html_kwargs, filter_column_colors, fixed_xaxis, val_ax_label, scatter_labels, frame_subset):
        
        self.data_filename = data_filename
        self.data_is_wide = data_is_wide
        self.out_filename = out_filename
        self.extension = self.get_extension()
        self.orientation = orientation
        self.sort = sort
        self.n_bars = n_bars
        self.fixed_order = fixed_order
        self.fixed_max = fixed_max
        self.steps_per_period = steps_per_period
        self.period_length = period_length
        self.end_period_pause = end_period_pause
        self.interpolate_period = interpolate_period
        self.period_label = self.get_period_label(period_label)
        self.period_template = period_template
        self.period_summary_func = period_summary_func
        self.perpendicular_bar_func = perpendicular_bar_func
        self.scatter_labels = scatter_labels
        #self.title = self.get_title(title)
        self.bar_size = bar_size
        self.bar_textposition = bar_textposition
        self.bar_texttemplate = self.get_bar_texttemplate(bar_texttemplate)
        self.scatter_texttemplate = self.get_scatter_texttemplate(scatter_texttemplate=None)
        self.bar_label_font = self.get_font(bar_label_font)
        self.tick_label_font = self.get_font(tick_label_font)
        self.hovertemplate = self.get_hovertemplate(hovertemplate)
        self.slider = slider
        self.scale = scale
        self.duration = self.period_length / steps_per_period
        self.write_html_kwargs = write_html_kwargs or {}
        self.filter_column_colors = filter_column_colors
        
        self.validate_params()
        self.bar_kwargs = self.get_bar_kwargs(bar_kwargs)
        self.layout_kwargs = self.get_layout_kwargs(layout_kwargs)
        self.df_vals, self.df_ranks, self.pw_names = self.get_plot_data()
        #self.col_filt = self.get_col_filt()
        self.pw_colors, pw_color_is_dark = self.generate_large_palette(len(self.pw_names))
        self.set_fixed_max_limits()
        self.str_index = self.df_vals.index.astype('str')

        self.bar_customdata = None
        self.bar_text = None
        self.insidetextfont = self.bar_label_font
        if not(self.scatter_labels):           
            self.insidetextfont = None
            self.bar_textposition = 'auto'
        #self.inside_label_font_colors = self.get_inside_label_colors(pw_color_is_dark)
        self.inside_label_font = {**self.bar_label_font, "color": "#696969"}
        self.outside_label_font = {**self.bar_label_font, "color": "#696969"}#"#2a2a2a"}

        self.fixed_xaxis = fixed_xaxis
        self.val_ax_range = self.get_val_ax_range(fixed_xaxis)
        self.val_ax_label = val_ax_label

        self.frame_subset = frame_subset

    def get_extension(self):
        if self.data_filename:
            return self.data_filename.split('.')[-1]

    def get_bar_texttemplate(self, bar_texttemplate):
        if bar_texttemplate is None:
            if self.scatter_labels:
                bar_texttemplate = "%{x:,.2f}"
            else:
                bar_texttemplate = "%{text}  %{x:,.2f}"#"%{x:,.2f}"#"%{customdata}  %{x:,.2f}"

        return bar_texttemplate


    def get_scatter_texttemplate(self, scatter_texttemplate):
        if scatter_texttemplate is None:
            scatter_texttemplate = "%{customdata}"
        return scatter_texttemplate

    def get_inside_label_colors(self, pw_color_is_dark):
        return np.array(["#ffffff" if val else "#2a2a2a" for val in pw_color_is_dark])

    def get_plot_area_pixels(self):
        width = 1000
        height = 800
        margin = self.layout_kwargs.get("margin", {})
        if not isinstance(margin, dict):
            margin = {}

        left = margin.get("l", 80)
        right = margin.get("r", 80)
        top = margin.get("t", 100)
        bottom = margin.get("b", 80)

        plot_width = max(1, width - left - right)
        plot_height = max(1, height - top - bottom)
        return plot_width, plot_height

    def get_value_axis_bounds(self, bar_vals):
        axis_range = self.xlimit if self.orientation == "h" else self.ylimit
        if axis_range is not None:
            axis_min, axis_max = axis_range
        else:
            max_val = float(np.nanmax(bar_vals)) if len(bar_vals) else 1.0
            min_val = float(np.nanmin(bar_vals)) if len(bar_vals) else 0.0
            axis_min = min(0.0, min_val)
            axis_max = max(1.0, max_val * 1.05 if max_val > 0 else 1.0)

        if axis_max <= axis_min:
            axis_max = axis_min + 1.0
        return axis_min, axis_max

    def safe_number_format(self, value, fmt):
        try:
            return format(value, fmt)
        except (ValueError, TypeError):
            return str(value)

    def render_label_text(self, name, x_val, y_val):
        template = self.bar_texttemplate or ""
        rendered = template.replace("%{customdata}", str(name))
        for axis_name, axis_value in (("x", x_val), ("y", y_val)):
            rendered = re.sub(
                rf"%\{{{axis_name}:([^}}]+)\}}",
                lambda m: self.safe_number_format(axis_value, m.group(1)),
                rendered
            )
            rendered = rendered.replace(f"%{{{axis_name}}}", str(axis_value))
        rendered = re.sub(r"%\{[^}]+\}", "", rendered)
        return rendered


    def labels_inside_bar(self, bar_vals):

        max_bar_val = max(bar_vals)

        return bar_vals >= max_bar_val / 2

    # def get_label_fit_mask(self, bar_vals, names, x_vals, y_vals):
    #     font_size = self.bar_label_font.get("size", 12)
    #     char_width_px = max(4.0, 0.62 * float(font_size))
    #     pad_px = 10.0

    #     plot_width, plot_height = self.get_plot_area_pixels()
    #     axis_min, axis_max = self.get_value_axis_bounds(bar_vals)
    #     axis_span = axis_max - axis_min

    #     rendered_texts = [
    #         self.render_label_text(name, x_val, y_val)
    #         for name, x_val, y_val in zip(names, x_vals, y_vals)
    #     ]
    #     text_widths_px = np.array(
    #         [(len(text) + 1) * char_width_px + pad_px for text in rendered_texts]
    #     )

    #     bar_lengths = np.maximum(np.asarray(bar_vals) - axis_min, 0) / axis_span
    #     bar_lengths_px = bar_lengths * plot_width

    #     return bar_lengths_px >= text_widths_px

    def validate_params(self):
        if isinstance(self.data_filename, str):
            if '.' not in self.data_filename:
                raise ValueError('`filename` must have an extension')
        elif self.data_filename is not None:
            raise TypeError('`filename` must be None or a string')
            
        if self.sort not in ('asc', 'desc'):
            raise ValueError('`sort` must be "asc" or "desc"')

        if self.orientation not in ('h', 'v'):
            raise ValueError('`orientation` must be "h" or "v"')

    def get_bar_kwargs(self, bar_kwargs):
        if bar_kwargs is None:
            return {'opacity': .8}
        elif isinstance(bar_kwargs, dict):
            if 'opacity' not in bar_kwargs:
                bar_kwargs['opacity'] = .8
            return bar_kwargs
        raise TypeError('`bar_kwargs` must be None or a dictionary mapping `go.Bar` parameters '
                        'to values.')

    def get_layout_kwargs(self, layout_kwargs):
        if layout_kwargs is None:
            return {'showlegend': False}
        elif isinstance(layout_kwargs, dict):
            if {'xaxis', 'yaxis', 'annotations'} & layout_kwargs.keys():
                raise ValueError('`layout_kwargs` cannot contain "xaxis", "yaxis", or '
                                 ' "annotations".')
            if 'showlegend' not in layout_kwargs:
                layout_kwargs['showlegend'] = False
            return layout_kwargs
        elif isinstance(layout_kwargs, plotly.graph_objs._layout.Layout):
            return self.get_layout_kwargs(layout_kwargs.to_plotly_json())
        raise TypeError('`layout_kwargs` must be None, a dictionary mapping '
                        '`go.Layout` parameters to values or an instance of `go.Layout`.')

    def get_val_ax_range(self, fixed_xaxis):
        if fixed_xaxis:
            val_ax_range = None
        else:
            val_ax_range = [0, self.df_vals.to_numpy().max()]
        return val_ax_range

    def get_period_label(self, period_label):
        if period_label is False:
            return False

        default_period_label = {'xref': 'paper', 'yref': 'paper', 'font': {'size': 20},
                                'xanchor': 'right', 'showarrow': False}
        if self.orientation == 'h':
            default_period_label['x'] = .95
            default_period_label['y'] = .15 if self.sort == 'desc' else .85
        else:
            default_period_label['x'] = .95 if self.sort == 'desc' else .05
            default_period_label['y'] = .85
            default_period_label['xanchor'] = 'left' if self.sort == 'asc' else 'right'

        if period_label is True:
            return default_period_label
        elif isinstance(period_label, dict):
            period_label = {**default_period_label, **period_label}
        else:
            raise TypeError('`period_label` must be a boolean or dictionary')

        return period_label

    def get_title(self, title):
        if title is None:
            return
        if isinstance(title, str):
            return {'text': title, 'y': 1, 'x': .5, 'xref': 'paper', 'yref': 'paper',
                    'pad': {'b': 10},
                    'xanchor': 'center', 'yanchor': 'bottom'}
        elif isinstance(title, (dict, plotly.graph_objects.layout.Title)):
            return title
        raise TypeError('`title` must be a string, dictionary, or '
                        '`plotly.graph_objects.layout.Title` instance')

    def get_font(self, font):
        if font is None:
            font = {'size': 12}
        elif isinstance(font, (int, float)):
            font = {'size': font}
        elif not isinstance(font, dict):
            raise TypeError('`font` must be a number or dictionary of font properties')
        return font

    def get_hovertemplate(self, hovertemplate):
        if hovertemplate is None:
            if self.orientation == 'h':
                return '%{y} - %{x:,.0f}<extra></extra>'
            return '%{x} - %{y:,.0f}<extra></extra>'
        return hovertemplate

    def get_pw_df_and_lut(self, pw_data_fpath):
        if os.path.exists('./data/pathway_data_idxed.csv') and os.path.exists('./data/pathway_names.txt'):
            df = pd.read_csv('./data/pathway_data_idxed.csv', index_col='global_index')
            with open('./data/pathway_names.txt') as f:
                pw_names = f.readlines()
            return df, pw_names

        df = pd.read_csv(pw_data_fpath, index_col='global_index')
        df = df.drop('p.adj', axis=1)
        pw_names = df["pathway.name"].unique()
        pw_idx_map = {name: i for i, name in enumerate(pw_names)}
        df["pathway.idx"] = df["pathway.name"].map(pw_idx_map)
        df = df.drop(columns=["pathway.name"])
        df.to_csv(os.path.join(os.path.dirname(pw_data_fpath),'pathway_data_idxed.csv'))
        with open(os.path.join(os.path.dirname(pw_data_fpath),'pathway_names.txt'), 'w') as f:
            for pw in pw_names: f.write(pw+'\n')
        return df, pw_names


    def get_wide_df_and_lut(self):
        if self.data_filename is None:
            self.data_filename = './data/pathway_data_wide.csv'
        if self.data_filename == './data/covid19.csv':
            df_wide = pd.read_csv(self.data_filename)
            return df_wide, None
        if os.path.exists(self.data_filename) and os.path.exists('./data/pathway_names.txt'):
            df_wide = pd.read_csv(self.data_filename, index_col='window')
            with open('./data/pathway_names.txt') as f:
                pw_names = np.array(f.readlines())
        else: 
            df, pw_names = self.get_pw_df_and_lut('./data/pathway_data.csv')
            df_wide = pd.pivot_table(df, values='-log10(p.adj)', index='window', columns='pathway.idx')
            df_wide = df_wide.fillna(0)
            df_wide.to_csv(self.data_filename)
        return df_wide, pw_names
    

    def get_plot_data(self):
        df_wide, pw_names = self.get_wide_df_and_lut()

        if self.n_bars is None:
            self.n_bars = (df_wide.iloc[0].values > 0).sum()
        else: 
            self.n_bars = min(self.n_bars, (df_wide.iloc[0].values > 0).sum())

        df_wide_idx = df_wide.index
        if df_wide.index[0] == 1:
            df_wide_idx -= 1

        df_wide.index = df_wide_idx * self.steps_per_period
        new_index = range(df_wide.index[-1]+1)
        df_wide = df_wide.reindex(new_index)

        df_wide_interp = df_wide.interpolate()
        topN = np.sort(df_wide_interp.to_numpy(), axis=1)[:, -self.n_bars:][:, ::-1]

        df_vals = pd.DataFrame(topN, index=df_wide_interp.index, columns=range(1, self.n_bars + 1))

        # get a dataframe with label (pathway) indices ranked after their value in each window
        df_ranks_wide = df_wide_interp.rank(axis=1, method='first', ascending=False)
        #df_ranks_wide = df_ranks_wide - (len(df_ranks_wide.columns)-self.n_bars)
        #print(df_ranks_wide.head(2))
        #df_ranks_wide[df_ranks_wide < 1] = np.nan
        df_ranks_wide[df_ranks_wide > self.n_bars] = np.nan
        ser = df_ranks_wide.stack().reset_index()

        df_ser = pd.DataFrame(ser).astype('int32')

        df_ranks = df_ser.pivot(index='window', columns=0, values='level_1')
        return df_vals, df_ranks, pw_names



    def _rgb_to_hex(self, r, g, b):
        return "#{:02X}{:02X}{:02X}".format(int(r * 255), int(g * 255), int(b * 255))


    def generate_large_palette(self, n,
                            sat_values=(0.9, 0.7, 0.55),
                            val_values=(0.95, 0.78),
                            hue_spacing_strategy='even'):
        """
        Generate n distinct colors by combining several saturation/value variations
        with a set of evenly spaced hues.
        - sat_values: tuple of saturation levels to use (0..1)
        - val_values: tuple of value/brightness levels to use (0..1)
        - hue_spacing_strategy: 'even' or 'golden' (golden reduces perceptual clustering)
        Returns list of hex strings length n.
        """
        variations = [(s, v) for v in val_values for s in sat_values]
        per_hue = len(variations)
        num_hues = math.ceil(n / per_hue)

        # generate hue list
        hues = []
        if hue_spacing_strategy == 'golden':
            golden = 0.618033988749895
            h = 0.0
            for i in range(num_hues):
                hues.append(h % 1.0)
                h += golden
        else:  # evenly spaced
            for i in range(num_hues):
                hues.append(i / float(num_hues))

        # interleave variations to avoid adjacent very-similar colors
        colors = []
        is_dark = []
        for i, h in enumerate(hues):
            # For variety, rotate variation order every hue
            for j in range(per_hue):
                s, v = variations[(j + i) % per_hue]
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                colors.append(self._rgb_to_hex(r, g, b))
                is_dark.append(np.sum(np.array([r, g, b] * np.array([299, 587, 114])))/1000 < 123) 
                # estimate perceived brightness of color for 
                #https://stackoverflow.com/questions/49437263/contrast-between-label-and-background-determine-if-color-is-light-or-dark
                if len(colors) == n:
                    perm = np.random.permutation(np.array(range(n)))
                    colors = np.array(colors)[perm]
                    is_dark = np.array(is_dark)[perm]
                    return colors, is_dark

    def set_fixed_max_limits(self):
        label_limit = (.2, self.n_bars + .8)
        value_limit = None
        min_val = 0 
        if self.fixed_max:
            value_limit = [min_val, self.df_vals.max().max() * 1.1]

        if self.orientation == 'h':
            self.xlimit = value_limit
            self.ylimit = label_limit
            

    def get_frames(self):
        frames = []
        slider_steps = []

        bar_locs = np.arange(self.n_bars, 0, -1)
        
        if not self.fixed_xaxis:
            max_bar_val = self.df_vals.to_numpy().max()

        if self.frame_subset is None: self.frame_subset = len(self.df_vals)
        for i in tqdm(range(len(self.df_vals[:self.frame_subset])), 'creating frames'):
            bar_vals_df = self.df_vals[self.df_vals.index==i]#self.df_vals.iloc[i, :self.n_bars].values

            data = self.get_data(i, bar_locs, bar_vals_df, max_bar_val)
            
            label_axis = dict()#dict(tickmode='array', tickvals=bar_locs, ticktext=None, 
                            # tickfont=self.tick_label_font)

            #label_axis['range'] = self.ylimit if self.orientation == 'h' else self.xlimit
            if self.orientation == 'v':
                label_axis['tickangle'] = -90

            value_axis = dict(showgrid=True, type=self.scale, title=self.val_ax_label)#, tickformat=',.0f')
            value_axis['range'] = self.xlimit if self.orientation == 'h' else self.ylimit

            xaxis, yaxis = (value_axis, label_axis) if self.orientation == 'h' \
                             else (label_axis, value_axis)
            
            annotations = self.get_annotations(i) 
            frame_name = str(i)
            if self.slider and i % self.steps_per_period == 0:
                slider_steps.append({
                    "args": [[frame_name],
                            {"frame": {"duration": self.duration, "redraw": False},
                            "mode": "immediate",
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                            }],
                    "label": self.get_period_label_text(i),
                    "method": "animate"
                })
            title_text = f'Locally enriched pathways per sliding window position {i/self.steps_per_period}'

            # Use go.Layout with title_text (canonical) so animations pick it up
            frame_layout = go.Layout(
                xaxis = xaxis,
                yaxis = label_axis, #annotations=annotations,
                autosize=False, width=1000, height=800,
                #margin = self.layout_kwargs.get('margin', {'l':150, 'r':80, 't':100, 'b':120}),
                title_text = title_text,
                **self.layout_kwargs
            )

            # name must be a string and slider steps must match the same string
            frames.append(go.Frame(data=data, layout=frame_layout, name=frame_name))

        return frames, slider_steps
    

    def get_data(self, i, bar_locs, bar_vals_df, max_bar_val):

        label_ids = self.df_ranks[self.df_ranks.index==i].values[0]#self.df_ranks.iloc[i].values
        x = bar_vals_df.transpose()[i].iloc[::-1]
        x.index = label_ids
        label_names = np.flip(self.pw_names[label_ids])
        y = pd.Series(label_names, index = label_ids)

        colors = self.pw_colors[label_ids]


        if self.fixed_xaxis:
            max_bar_val = max(x.values)

        #bar_locs = bar_locs + np.random.rand(len(bar_locs)) / 10000 # done to prevent stacking of bars
        #x, y = (bar_vals, bar_locs) if self.orientation == 'h' else (bar_locs, bar_vals)

        if not(self.scatter_labels): 
            val_labels = np.round(x, 2).astype(str)
            #self.bar_customdata = self.pw_names[label_ids]
            self.bar_text=np.char.add(self.pw_names[label_ids], np.char.add(['  '], val_labels))

        bar = go.Bar(
            x=x, y=y,
            #ids=label_ids.astype(str),
            #customdata=self.bar_customdata,
            #text = self.bar_text,
            #textposition=self.bar_textposition,
            hoverinfo='all',
            texttemplate='%{y} %{x:.4s}',#self.bar_texttemplate,
            textangle=0,            
            textposition='auto',#outside',
            orientation=self.orientation,
            marker_color=colors,
            cliponaxis=False,
            #insidetextfont=self.insidetextfont,
            #outsidetextfont=self.bar_label_font,
            #hovertemplate=self.hovertemplate,
            **self.bar_kwargs
        )     

        if self.scatter_labels: 

            labels_inside_bar = x.values >= max_bar_val / 2
            inside_offset = 0.015*max_bar_val
            outside_offsets = 0.05*max_bar_val#[0.01*len(str(bar_val)) for bar_val in bar_vals]
            inside_x = np.asarray(x - inside_offset, dtype=object)
            inside_y = np.asarray(y, dtype=object)
            outside_x = np.asarray(x + outside_offsets, dtype=object)#np.asarray(x + outside_offset, dtype=object)
            outside_y = np.asarray(y, dtype=object)

            inside_x[~labels_inside_bar] = None
            inside_y[~labels_inside_bar] = None
            outside_x[labels_inside_bar] = None
            outside_y[labels_inside_bar] = None

            #self.inside_label_font = {**self.bar_label_font, "color": self.inside_label_font_colors[label_ids[labels_inside_bar]]}

            inside_labels = go.Scatter(
                x=inside_x, y=inside_y,
                ids=label_ids.astype(str),
                customdata=label_names,
                mode="text",
                texttemplate=self.scatter_texttemplate,
                textposition="middle left",
                textfont=self.inside_label_font,
                cliponaxis=False,
                hoverinfo="skip",
                showlegend=False
            )

            outside_labels = go.Scatter(
                x=outside_x, y=outside_y,
                ids=label_ids.astype(str),
                customdata=label_names,
                mode="text",
                texttemplate=self.scatter_texttemplate,
                textposition="middle right",
                textfont=self.outside_label_font,
                cliponaxis=False,
                hoverinfo="skip",
                showlegend=False
            )

            return [bar, inside_labels, outside_labels]
        
        else:
            return [bar]

    def get_period_label_text(self, i):
        if self.period_template:
            idx_val = self.df_vals.index[i]
            if self.df_vals.index.dtype.kind == 'M':
                s = idx_val.strftime(self.period_template)
            else:
                s = self.period_template.format(x=idx_val)
        else:
            s = self.str_index[i]
        return s
    
    def get_annotations(self, i):
        annotations = []
        if self.period_label:
            #self.period_label['text'] = self.get_period_label_text(i) + ' ' + str(i)
            pos = i/self.steps_per_period
            self.period_label['text'] = 'pos.:' + ' ' + str(pos)
            annotations.append(self.period_label)

        if self.period_summary_func:
            values = self.df_vals.iloc[i]
            ranks = self.df_ranks.iloc[i]
            text_dict = self.period_summary_func(values, ranks)
            if 'x' not in text_dict or 'y' not in text_dict or 'text' not in text_dict:
                name = self.period_summary_func.__name__
                raise ValueError(f'The dictionary returned from `{name}` must contain '
                                  '"x", "y", and "s"')
            text, x, y = text_dict['text'], text_dict['x'], text_dict['y']
            annotations.append(dict(text=text, x=x, y=y, font=dict(size=14), 
                                    xref="paper", yref="paper", showarrow=False))

        return annotations
    

    def make_animation(self):
        frames, slider_steps = self.get_frames()
        data = frames[0].data
        layout = frames[0].layout
        layout.updatemenus = [dict(
            type="buttons",
            direction = "left",
            x=1, 
            y=1.02,
            xanchor='right',
            yanchor='bottom',
            buttons=[dict(label="Play",
                          method="animate",
                          # redraw must be true for bar plots
                          args=[None, {"frame": {"duration": self.duration, "redraw": False},#{"duration": 500, "redraw": False},#
                                        "fromcurrent": True,
                                        "mode": "immediate",
                                        "transition": {"duration": self.duration, "easing": "linear"}#{"duration": 300, "easing": "quadratic-in-out"}
                                    }]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}]),
                     ]
                     )]

        sliders_dict = {
                        "active": 0,
                        "yanchor": "top",
                        "xanchor": "left",
                        "currentvalue": {
                            # "font": {"size": 20},
                            # "prefix": '', # allow user to set
                            "visible": False, # just repeats period label
                            # "xanchor": "right"
                        },
                        "transition": {"duration": self.duration, "easing": "cubic-in-out"},#{"duration": 300, "easing": "cubic-in-out"}, #transition duration must be set at least as long as frame duration
                        "pad": {"b": 10, "t": 50},
                        "len": 1,
                        "x": 0,
                        "y": 0,
                        "steps": slider_steps
                    }
        if self.slider:
            layout.sliders = [sliders_dict]

        fig = go.Figure(data=data, layout=layout, frames=frames)
        fig.update_yaxes(title_text = f'Top {self.n_bars} pathways', visible = True, showticklabels= False)
        if self.out_filename:
            fig.write_html(self.out_filename, **self.write_html_kwargs)
        else:
            return fig


def bar_chart_race_plotly(data_filename=None, data_is_wide=False, out_filename=None, orientation='h', sort='desc', n_bars=None, 
                          fixed_order=False, fixed_max=False, steps_per_period=10, 
                          period_length=500, end_period_pause=0, interpolate_period=True, 
                          period_label=True, period_template=None, period_summary_func=None, 
                          perpendicular_bar_func=None, title=None, bar_size=.95, 
                          bar_textposition='outside', bar_texttemplate=None, bar_label_font=None, 
                          tick_label_font=None, hovertemplate=None, slider=True, scale='linear', 
                          bar_kwargs=None, layout_kwargs=None, write_html_kwargs=None, 
                          filter_column_colors=False, fixed_xaxis = False, val_ax_label=None, scatter_labels=True, frame_subset=None):
    '''
    Create an animated bar chart race using Plotly. Data must be in 
    'wide' format where each row represents a single time period and each 
    column represents a distinct category. Optionally, the index can label 
    the time period. Bar length and location change linearly from one time 
    period to the next.

    Note - The duration of each frame is calculated as 
    `period_length` / `steps_per_period`, but is unlikely to actually 
    be this number, especially when duration is low (< 50ms). You may have to
    experiment with different combinations of `period_length` and
    `steps_per_period` to get the animation at the desired speed.

    If no `filename` is given, a plotly figure is returned that is embedded
    into the notebook.

    Parameters
    ----------
    df : pandas DataFrame
        Must be a 'wide' DataFrame where each row represents a single period 
        of time. Each column contains the values of the bars for that 
        category. Optionally, use the index to label each time period.
        The index can be of any type.

    filename : `None` or str, default None
        If `None` return plotly animation, otherwise save
        to disk. Can only save as HTML at this time.

    orientation : 'h' or 'v', default 'h'
        Bar orientation - horizontal or vertical

    sort : 'desc' or 'asc', default 'desc'
        Choose how to sort the bars. Use 'desc' to put largest bars on top 
        and 'asc' to place largest bars on bottom.

    n_bars : int, default None
        Choose the maximum number of bars to display on the graph. 
        By default, use all bars. New bars entering the race will appear 
        from the edge of the axes.

    fixed_order : bool or list, default False
        When `False`, bar order changes every time period to correspond 
        with `sort`. When `True`, bars remained fixed according to their 
        final value corresponding with `sort`. Otherwise, provide a list 
        of the exact order of the categories for tiod to the next. 
        The bars will grow linearly between each period.

    period_length : int, default 500
        Number of milliseconds to animate each period (row). 
        Default is 500ms (half of a second)

    end_period_pause : int, default 0
        Number of milliseconds to pause the animation at the end of
        each period.

    interpolate_period : bool, default `False`
        Whether to interpolate the period. Only valid for datetime or
        numeric indexes. When set to `True`, for example, 
        the two consecutive periods 2020-03-29 and 2020-03-30 with 
        `steps_per_period` set to 4 would yield a new index of
        2020-03-29 00:00:00
        2020-03-29 06:00:00
        2020-03-29 12:00:00
        2020-03-29 18:00:00
        2020-03-30 00:00:00
    
    period_label : bool or dict, default `True`
        If `True` or dict, use the index as a large text label
        on the figure labeling each period. No label when 'False'.

        Use a dictionary to supply the exact position of the period
        along with any valid parameters of a plotly annotation.

        Example:
        {
            'x': .99,
            'y': .8,
            'font' : {'family': 'Helvetica', 'size': 20, 'color': 'orange'},
            'xanchor': 'right',
        }
        
        Reference - https://plotly.com/python/reference/#layout-annotations

        The default location depends on `orientation` and `sort`
        * h, desc -> x=.95, y=.15
        * h, asc -> x=.95, y=.85
        * v, desc -> x=.95, y=.85
        * v, asc -> x=.05, y=.85

    period_template : str, default `None`
        Either a string with date directives or 
        a new-style (Python 3.6+) formatted string

        For a string with a date directive, find the complete list here
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
        
        Example of string with date directives
            '%B %d, %Y'
        Will change 2020/03/29 to March 29, 2020
        
        For new-style formatted string. Use curly braces and the variable `x`, 
        which will be passed the current period's index value.
        Example:
            'Period {x:10.2f}'

        Date directives will only be used for datetime indexes.

    period_summary_func : function, default None
        Custom text added to the axes each period.
        Create a user-defined function that accepts two pandas Series of the 
        current time period's values and ranks. It must return a dictionary 
        containing at a minimum the keys "x", "y", and "text" which will be 
        passed used for a plotly annotation.

        Example:
        def func(values, ranks):
            total = values.sum()
            text = f'Worldwide deaths: {total}'
            return {'x': .85, 'y': .2, 'text': text, 'size': 11}

    perpendicular_bar_func : function or str, default None
        Creates a single bar perpendicular to the main bars that spans the 
        length of the axis. 
        
        Use either a string that the DataFrame `agg` method understands or a 
        user-defined function.
            
        DataFrame strings - 'mean', 'median', 'max', 'min', etc..

        The function is passed two pandas Series of the current time period's
        data and ranks. It must return a single value.

        def func(values, ranks):
            return values.quantile(.75)

    colors : str or sequence colors, default 'dark12'
        Colors to be used for the bars. All matplotlib and plotly colormaps are 
        available by string name. Colors will repeat if there are more bars than colors.

        'dark12' is the default colormap. If there are more than 10 columns, 
        then the default colormap will be 'dark24'

        Append "_r" to the colormap name to use the reverse of the colormap.
        i.e. "dark12_r"

    title : str, dict, or plotly.graph_objects.layout.Title , default None
        Title of animation. Use a string for simple titles or a
        dictionary to specify several properties
        {'text': 'My Bar Chart Race', 
         'x':0.5, 
         'y':.9,
         'xanchor': 'center', 
         'yanchor': 'bottom'}

        Other properties include: font, pad, xref, yref

    bar_size : float, default .95
        Height/width of bars for horizontal/vertical bar charts. 
        Use a number between 0 and 1
        Represents the fraction of space that each bar takes up. 
        When equal to 1, no gap remains between the bars.

    bar_textposition : str or sequence, default `None`
        Position on bar to place its label.
        Use one of the strings - 'inside', 'outside', 'auto', 'none'
        or a sequence of the above

    bar_texttemplate : str, default '%{x:,.0f}' or '%{y:,.0f}'
        Template string used for rendering the text inside/outside
        the bars. Variables are inserted using %{variable},
        for example "y: %{y}". Numbers are formatted using
        d3-format's syntax %{variable:d3-format}, for example
        "Price: %{y:$.2f}".

    bar_label_font : number or dict, None
        Font size of numeric bar labels. When None, font size is 12. 
        Use a dictionary to supply several font properties.
        Example:
        {
            'size': 12,
            'family': 'Courier New, monospace',
            'color': '#7f7f7f'
        }

    tick_label_font : number or dict, None
        Font size of tick labels.When None, font size is 12. 
        Use a dictionary to supply several font properties.

    hovertemplate : str, default None
        Template string used for rendering the information that appear 
        on hover box. By default, it is '%{y} - %{x:,.0f}<extra></extra>'

        Reference: https://plotly.com/python/hover-text-and-formatting

    slider : bool, default True
        Whether or not to place a slider below the animation

    scale : 'linear' or 'log', default 'linear'
        Type of scaling to use for the axis containing the values

    bar_kwargs : dict, default `None` (opacity=.8)
        Other keyword arguments (within a dictionary) forwarded to the 
        plotly `go.Bar` function. If no value for 'opacity' is given,
        then it is set to .8 by default.

    layout_kwargs : dict or go.Layout instance, default None
        Other keyword arguments (within a dictionary) are forwarded to 
        the plotly `go.Layout` function. Use this to control the size of
        the figure.
        Example:
        {
            'width': 600,
            'height': 400,
            'showlegend': True
        }

    write_html_kwargs : dict, default None
        Arguments passed to the write_html plotly go.Figure method.
        Example:
        {
            'auto_play': False,
            'include_plotlyjs': 'cdn',
            'full_html': False=
        }
        Reference: https://plotly.github.io/plotly.py-docs/generated/plotly.io.write_html.html
                   
    filter_column_colors : bool, default `False`
        When setting n_bars, it's possible that some columns never 
        appear in the animation. Regardless, all columns get assigned
        a color by default. 
        
        For instance, suppose you have 100 columns 
        in your DataFrame, set n_bars to 10, and 15 different columns 
        make at least one appearance in the animation. Even if your 
        colormap has at least 15 colors, it's possible that many 
        bars will be the same color, since each of the 100 columns is
        assigned of the colormaps colors.

        Setting this to `True` will map your colormap to just those 
        columns that make an appearance in the animation, helping
        avoid duplication of colors.

        Setting this to `True` will also have the (possibly unintended)
        consequence of changing the colors of each color every time a 
        new integer for n_bars is used.

        EXPERIMENTAL
        This parameter is experimental and may be changed/removed
        in a later version.

    Returns
    -------
    When `filename` is left as `None`, a plotly figure is returned and
    embedded into the notebook. Otherwise, a file of the HTML is 
    saved and `None` is returned.

    References
    -----
    Plotly Figure - https://plotly.com/python/reference
    Plotly API - https://plotly.com/python-api-reference
    d3 formatting - https://github.com/d3/d3-3.x-api-reference/blob/master/Formatting.md
    
    Examples
    --------
    Use the `load_data` function to get an example dataset to 
    create an animation.

    df = bcr.load_dataset('covid19')
    bcr.bar_chart_race_plotly(
        df=df, 
        filename='covid19_horiz_desc.html', 
        orientation='h', 
        sort='desc', 
        n_bars=8, 
        fixed_order=False, 
        fixed_max=True, 
        steps_per_period=10, 
        period_length=500, 
        interpolate_period=False, 
        period_label={'x': .99, 'y': .8, 'font': {'size': 25, 'color': 'blue'}}, 
        period_template='%B %d, %Y', 
        period_summary_func=lambda v, r: {'x': .85, 'y': .2, 
                                          's': f'Total deaths: {v.sum()}', 
                                          'size': 11}, 
        perpendicular_bar_func='median', 
        colors='dark12', 
        title='COVID-19 Deaths by Country', 
        bar_size=.95,
        bar_textposition='outside', 
        bar_texttemplate='%{x}',
        bar_label_font=12, 
        tick_label_font=12, 
        hovertemplate=None,
        scale='linear', 
        bar_kwargs={'opacity': .7},
        write_html_kwargs=None,
        filter_column_colors=False)        
    '''
    bcr = _BarChartRace(data_filename, data_is_wide, out_filename, orientation, sort, n_bars, fixed_order, fixed_max,
                        steps_per_period, period_length, end_period_pause, interpolate_period, 
                        period_label, period_template, period_summary_func, perpendicular_bar_func, 
                        title, bar_size, bar_textposition, bar_texttemplate, bar_label_font, 
                        tick_label_font, hovertemplate, slider, scale, bar_kwargs, layout_kwargs, 
                        write_html_kwargs, filter_column_colors, fixed_xaxis, val_ax_label, scatter_labels, frame_subset)
    return bcr
