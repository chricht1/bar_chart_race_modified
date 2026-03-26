import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly
import os
from tqdm import tqdm
import colorsys
import math
import re



class _BarChartRace:
    
    def __init__(self, data_filename=None, data_is_wide=False, out_filename=None, orientation='h', sort='desc', n_bars=None, 
                          fixed_order=False, fixed_max=False, steps_per_period=10, 
                          period_length=500, end_period_pause=0, interpolate_period=True, 
                          period_label=True, period_template=None, period_summary_func=None, 
                          perpendicular_bar_func=None, title=None, bar_size=.95, 
                          bar_textposition='outside', bar_texttemplate=None, bar_label_font=None, 
                          tick_label_font=None, hovertemplate=None, slider=True, slider_dict=None, scale='linear', 
                          bar_kwargs=None, layout_kwargs=None, write_html_kwargs=None, 
                          filter_column_colors=False, fixed_xaxis = True, xaxis_label='', yaxis_label='', show_yaxis_ticklabels=True,
                          layout_font_size=10, resolution_scale=1.0, scale_fonts=True,
                          scatter_labels=True, frame_subset=None, plot_labels_over_bars=False, labels_max_len=None):
        
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
        self.bar_size = bar_size
        self.bar_textposition = bar_textposition
        self.bar_texttemplate = self.get_bar_texttemplate(bar_texttemplate)
        self.scatter_texttemplate = self.get_scatter_texttemplate(scatter_texttemplate=None)
        self.bar_label_font = self.get_font(bar_label_font)
        self.tick_label_font = self.get_font(tick_label_font)
        self.hovertemplate = self.get_hovertemplate(hovertemplate)
        self.slider = slider
        self.slider_dict = slider_dict
        self.scale = scale
        self.duration = self.period_length / steps_per_period
        self.write_html_kwargs = write_html_kwargs or {}
        self.filter_column_colors = filter_column_colors
        self.resolution_scale = resolution_scale
        self.scale_fonts = scale_fonts
        
        self.validate_params()
        self.bar_kwargs = self.get_bar_kwargs(bar_kwargs)
        self.layout_kwargs = self.get_layout_kwargs(layout_kwargs)
        self.layout_kwargs = self.scale_layout_kwargs(self.layout_kwargs)
        self.period_label = self.scale_period_label(self.period_label)
        self.bar_label_font = self.scale_font_dict(self.bar_label_font)
        self.tick_label_font = self.scale_font_dict(self.tick_label_font)
        self.slider_dict = self.scale_slider_dict(self.slider_dict)
        self.df_vals, self.df_ranks, self.pw_names = self.get_plot_data()
        self.frame_subset = frame_subset
        if self.frame_subset is None: self.frame_subset = [0,len(self.df_vals)]
        if self.frame_subset[1] is None: self.frame_subset = [frame_subset[0],len(self.df_vals)]
        self.col_filt = self.get_col_filt()
        self.pw_colors, self.pw_color_is_dark = self.generate_large_palette(len(self.pw_names))
        self.set_fixed_max_limits()
        self.str_index = self.df_vals.index.astype('str')

        self.bar_customdata = None
        self.bar_text = None
        self.insidetextfont = self.bar_label_font
        if not(self.scatter_labels):           
            self.insidetextfont = None
            self.bar_textposition = 'auto'
        self.inside_label_font = {**self.bar_label_font, "color": self.get_inside_label_colors(self.pw_color_is_dark)}
        self.outside_label_font = {**self.bar_label_font, "color": "#2a2a2a"}#"#696969"

        self.fixed_xaxis = fixed_xaxis
        self.val_ax_range = self.get_val_ax_range(fixed_xaxis)
        self.xaxis_label = xaxis_label
        self.yaxis_label = yaxis_label
        self.show_yaxis_ticklabels = show_yaxis_ticklabels
        self.layout_font_size = self.scale_font_size(layout_font_size)

        self.labels_max_len = labels_max_len
        if self.labels_max_len is not None: self.shorten_pw_names()
        self.plot_labels_over_bars = plot_labels_over_bars
        self.title = self.scale_title(self.get_title(title))
        self.layout_height = self.get_layout_params()

    def get_layout_params(self):
        #if self.title_text is None:
        #    self.title_text = f'Fold-change-specific enriched pathways per sliding window position'
        
        layout_height = 800
        if self.plot_labels_over_bars:
            layout_height += self.n_bars*10
        return layout_height

    def shorten_pw_names(self):
        for i, pw in enumerate(self.pw_names):
            if len(pw) > self.labels_max_len:
                self.pw_names[i] = self.pw_names[i][:self.labels_max_len-3] + '...' 

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
        width = self.layout_kwargs.get("width", int(1000 * self.resolution_scale))
        height = self.layout_kwargs.get("height", int(800 * self.resolution_scale))
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

    def scale_font_size(self, size):
        if not self.scale_fonts or self.resolution_scale == 1:
            return size
        if isinstance(size, (int, float)):
            return size * self.resolution_scale
        return size

    def scale_font_dict(self, font_dict):
        if not self.scale_fonts or self.resolution_scale == 1:
            return font_dict
        if not isinstance(font_dict, dict):
            return font_dict
        scaled = dict(font_dict)
        if "size" in scaled and isinstance(scaled["size"], (int, float)):
            scaled["size"] = scaled["size"] * self.resolution_scale
        return scaled

    def scale_period_label(self, period_label):
        if not self.scale_fonts or self.resolution_scale == 1:
            return period_label
        if period_label is False or period_label is None:
            return period_label
        if isinstance(period_label, dict):
            scaled = dict(period_label)
            font = scaled.get("font")
            if isinstance(font, dict):
                scaled["font"] = self.scale_font_dict(font)
            return scaled
        return period_label

    def scale_title(self, title):
        if not self.scale_fonts or self.resolution_scale == 1:
            return title
        if isinstance(title, dict):
            scaled = dict(title)
            font = scaled.get("font")
            if isinstance(font, dict):
                scaled["font"] = self.scale_font_dict(font)
            return scaled
        return title

    def scale_slider_dict(self, slider_dict):
        if not self.scale_fonts or self.resolution_scale == 1:
            return slider_dict
        if not isinstance(slider_dict, dict):
            return slider_dict
        scaled = dict(slider_dict)
        currentvalue = scaled.get("currentvalue")
        if isinstance(currentvalue, dict):
            cv = dict(currentvalue)
            font = cv.get("font")
            if isinstance(font, dict):
                cv["font"] = self.scale_font_dict(font)
            scaled["currentvalue"] = cv
        if "font" in scaled and isinstance(scaled["font"], dict):
            scaled["font"] = self.scale_font_dict(scaled["font"])
        pad = scaled.get("pad")
        if isinstance(pad, dict):
            pad_scaled = dict(pad)
            for key in ("b", "t", "l", "r"):
                if key in pad_scaled and isinstance(pad_scaled[key], (int, float)):
                    pad_scaled[key] = pad_scaled[key] * self.resolution_scale
            scaled["pad"] = pad_scaled
        return scaled

    def scale_layout_kwargs(self, layout_kwargs):
        if not isinstance(layout_kwargs, dict) or self.resolution_scale == 1:
            return layout_kwargs
        scaled = dict(layout_kwargs)
        for key in ("width", "height"):
            if key in scaled and isinstance(scaled[key], (int, float)):
                scaled[key] = scaled[key] * self.resolution_scale
        margin = scaled.get("margin")
        if isinstance(margin, dict):
            margin_scaled = dict(margin)
            for key in ("l", "r", "t", "b"):
                if key in margin_scaled and isinstance(margin_scaled[key], (int, float)):
                    margin_scaled[key] = margin_scaled[key] * self.resolution_scale
            scaled["margin"] = margin_scaled
        return scaled

    def get_value_axis_bounds(self, bar_vals):
        axis_range = self.xlimit if self.orientation == "h" else self.ylimit
        if axis_range is not None:
            axis_min, axis_max = axis_range
        else:
            max_val = float(np.nanmax(bar_vals)) if len(bar_vals) else 1.0
            min_val = float(np.nanmin(bar_vals)) if len(bar_vals) else 0.0
            axis_min = min(0.0, min_val)
            axis_max = max(1.0, max_val * 1.1 if max_val > 0 else 1.0)

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

        if not isinstance(self.resolution_scale, (int, float)) or self.resolution_scale <= 0:
            raise ValueError('`resolution_scale` must be a positive number')
        if not isinstance(self.scale_fonts, bool):
            raise TypeError('`scale_fonts` must be a boolean')

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
            return {'text': title, 'x': 0.5, 'y':0.91, 'xref': 'paper', 'yref': 'paper',
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
        #df_vals.to_csv('df_vals_interp.csv')

        df_ranks_wide = df_wide_interp.rank(axis=1, method='first', ascending=False)
        df_ranks_wide[df_ranks_wide > self.n_bars] = np.nan

        ser = df_ranks_wide.stack().reset_index()

        df_ser = pd.DataFrame(ser).astype('int32')

        df_ranks = df_ser.pivot(index=df_ser.columns[0], columns=0, values=df_ser.columns[1])
        return df_vals, df_ranks, pw_names


    def get_col_filt(self):
        col_filt = pd.Series([True] * self.df_vals.shape[1])
        if self.n_bars < self.df_ranks.shape[1]:
            orient_sort = self.orientation, self.sort
            if orient_sort in [('h', 'asc'), ('v', 'desc')]:
                # 1 is high
                col_filt = (self.df_ranks < self.n_bars + .99).any()
            else:
                # 1 is low
                col_filt = (self.df_ranks > 0).any()

            if self.filter_column_colors and not col_filt.all():
                self.df_vals = self.df_vals.loc[:, col_filt]
                self.df_ranks = self.df_ranks.loc[:, col_filt]
        return col_filt
        

    #def compute_():

    def _rgb_to_hex(self, r, g, b):
        return "#{:02X}{:02X}{:02X}".format(r, g, b)


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
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
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
        label_limit = (.5, self.n_bars + .5)
        value_limit = None
        min_val = 0 
        if self.fixed_max:
            value_limit = [min_val, self.df_vals[self.frame_subset[0]:self.frame_subset[1]].max().max() * 1.6]
        
        self.xlimit = value_limit
        self.ylimit = label_limit
            

    def get_frames(self):
        frames = []
        slider_steps = []

        bar_locs = np.arange(self.n_bars, 0, -1)


        for i in tqdm(range(self.frame_subset[0],self.frame_subset[1]), 'creating frames'):
            
            bar_vals = self.df_vals.iloc[i, :self.n_bars].values

            data = self.get_data(i, bar_locs, bar_vals)

            label_axis = dict(title_text = self.yaxis_label, range=self.ylimit, showticklabels=self.show_yaxis_ticklabels, tickfont=self.tick_label_font)#dict(tickmode='array', tickvals=bar_locs, ticktext=None, 

            value_axis = dict(showgrid=True, type=self.scale, title=self.xaxis_label, tickfont=self.tick_label_font)#tickformat=',.0f')
            value_axis['range'] = self.xlimit
            
            xaxis, yaxis = value_axis, label_axis
            
            #annotations = self.get_annotations(i) 
            if self.slider and i % self.steps_per_period == 0:
                slider_steps.append(
                            {"args": [[i],
                                {"frame": {"duration": self.duration, "redraw": False},
                                 "mode": "immediate",
                                 "fromcurrent": True,
                                 "transition": {"duration": 0}#self.duration}
                                }],
                            "label": self.get_period_label_text(i), 
                            "method": "animate"})
                
            yaxis['title_standoff'] = yaxis.get('title_standoff', 10)
            layout = go.Layout(xaxis=xaxis, yaxis=yaxis, #annotations=annotations,
                                autosize=False, #margin={'l': 150}, 
                                title=self.title, font=dict(size=self.layout_font_size),
                                **self.layout_kwargs)
            #if self.perpendicular_bar_func:
                #pbar = self.get_perpendicular_bar(bar_vals, i, layout)
                #layout.update(shapes=[pbar], overwrite=True)
            frames.append(go.Frame(data=data, layout=layout, name=i))

        return frames, slider_steps
    

    def get_data(self, i, bar_locs, bar_vals):

        label_ids = self.df_ranks.iloc[i].values
        colors = self.pw_colors[label_ids]#pd.Series(self.pw_colors)#
        label_names = self.pw_names[label_ids]

        if not self.fixed_xaxis:
            val_ax_max = max(bar_vals)
        else: 
            val_ax_max = self.xlimit[1]

        #bar_locs = bar_locs + np.random.rand(len(bar_locs)) / 10000 # done to prevent stacking of bars
        x, y = (bar_vals, bar_locs) if self.orientation == 'h' else (bar_locs, bar_vals)
        x[x==0] = np.nan

        if not(self.scatter_labels): 
            val_labels = np.round(x, 2).astype(str)
            #self.bar_customdata = self.pw_names[label_ids]
            self.bar_text=np.char.add(self.pw_names[label_ids], np.char.add(['  '], val_labels))

        #print('x: ', x)
        #print('y: ', y)

        bar = go.Bar(
            x=x, y=y,
            ids=label_ids.astype(str), # needed for bar switching animation
            #customdata=self.bar_customdata,
            textposition=self.bar_textposition,
            texttemplate='%{x:.2f}',#self.bar_texttemplate,
            marker_color=colors,
            cliponaxis=False,
            insidetextfont=self.insidetextfont,
            outsidetextfont=self.bar_label_font,
            orientation = 'h',
            hovertemplate=self.hovertemplate,
            **self.bar_kwargs
        )     

        if self.scatter_labels: 
            label_names[x==0] = ''

            labels_inside_bar = bar_vals >= val_ax_max / 2
            inside_offset = 0.015*val_ax_max # 0.015
            outside_offset = 0.015*val_ax_max+0.9*val_ax_max/15 # 0.05
            inside_x = np.asarray(x - inside_offset, dtype=object)
            inside_y = np.asarray(y, dtype=object)
            outside_x = np.asarray(x + outside_offset, dtype=object)#np.asarray(x + outside_offset, dtype=object)
            outside_y = np.asarray(y, dtype=object)

            inside_x[~labels_inside_bar] = None
            inside_y[~labels_inside_bar] = None
            outside_x[labels_inside_bar] = None
            outside_y[labels_inside_bar] = None

            self.inside_label_font = {**self.bar_label_font, "color": self.get_inside_label_colors(self.pw_color_is_dark[label_ids])}

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

    def get_perpendicular_bar(self, bar_vals, i, layout):
        if isinstance(self.perpendicular_bar_func, str):
            val = pd.Series(bar_vals).agg(self.perpendicular_bar_func)
        else:
            values = self.df_vals.iloc[i]
            ranks = self.df_ranks.iloc[i]
            val = self.perpendicular_bar_func(values, ranks)

        xref, yref = ("x", "paper") if self.orientation == 'h' else ("paper", "y")
        value_limit = self.xlimit if self.orientation == 'h' else self.ylimit
        if self.fixed_max:
            delta = (value_limit[1] - value_limit[0]) * .02
        else:
            delta = (1.05 * bar_vals.max() - bar_vals.min()) * .02

        x0, x1 = (val - delta, val + delta) if self.orientation == 'h' else (0, 1)
        y0, y1 = (val - delta, val + delta) if self.orientation == 'v' else (0, 1)

        return dict(type="rect", xref=xref, yref=yref, x0=x0, y0=y0, x1=x1, y1=y1,
                    fillcolor="#444444",layer="below", opacity=.5, line_width=0)

    def make_animation(self):
        frames, slider_steps = self.get_frames()
        data = frames[0].data
        layout = frames[0].layout
        layout.updatemenus = [dict(
            type="buttons",
            direction = "left",
            x=0,
            y=-0.07, #-0.05 #(self.slider_dict["font"]/1000)*2.2
            xanchor='left',
            yanchor='top',
            buttons=[dict(label="Play",
                          method="animate",
                          # redraw must be true for bar plots
                          args=[None, {"frame": {"duration": 500, "redraw": False},#{"duration": self.duration, "redraw": False},
                                        "fromcurrent": True,
                                        "mode": "immediate",
                                        "transition": {"duration": 300, "easing": "quadratic-in-out"}#{"duration": 0, "easing": "linear"}
                                    }]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}]),
                     ]
                     )]

        if self.slider_dict is None:
            self.slider_dict = {
                        "active": 0,
                        "yanchor": "top",
                        "xanchor": "left",
                        "currentvalue": {
                            "font": {"size": 23},
                            "prefix": "Position: ",
                            "visible": True,
                            "xanchor": "right"
                        },
                        "pad": {"b": 10, "t": 20},
                        "len": 1,
                        "x": 0,
                        "y": -0.04
                    }
            self.slider_dict = self.scale_slider_dict(self.slider_dict)
        self.slider_dict["transition"] = {"duration": self.duration, "easing": "cubic-in-out"} #transition duration must be set at least as long as frame duration
        self.slider_dict["steps"] = slider_steps
        if self.slider:
            layout.sliders = [self.slider_dict]

        fig = go.Figure(data=data, layout=layout, frames=frames[1:])
        if self.out_filename:
            fig.write_html(self.out_filename, **self.write_html_kwargs)
        return fig
