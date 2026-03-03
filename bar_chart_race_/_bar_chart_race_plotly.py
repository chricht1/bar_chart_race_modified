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
    
    def __init__(self, data_filename=None, data_is_wide=False, out_filename=None, n_bars=None, 
                          fixed_order=False, fixed_max=False, steps_per_period=10, 
                          period_length=500, end_period_pause=0, interpolate_period=True, 
                          perpendicular_bar_func=None, title=None, bar_size=.95, 
                          bar_textposition='outside', bar_texttemplate=None, bar_label_font=None, 
                          tick_label_font=None, hovertemplate=None, slider=True, scale='linear', 
                          write_html_kwargs=None,
                          fixed_xaxis = False, plot_pws_yaxis=False, val_ax_label=None, 
                          scatter_labels=False, scatter_values=False, linebreak_labels=False, 
                          labels_max_len=None, frame_subset=None):
        
        self.data_filename = data_filename
        self.data_is_wide = data_is_wide
        self.out_filename = out_filename
        
        self.fixed_xaxis = fixed_xaxis
        self.plot_pws_yaxis = plot_pws_yaxis
        self.val_ax_range = None
        self.val_ax_label = val_ax_label

        self.frame_subset = frame_subset
        self.n_bars = n_bars
        self.fixed_order = fixed_order
        self.fixed_max = fixed_max
        self.steps_per_period = steps_per_period
        self.period_length = period_length
        self.end_period_pause = end_period_pause
        self.interpolate_period = interpolate_period
        self.perpendicular_bar_func = perpendicular_bar_func
        self.scatter_labels = scatter_labels
        self.scatter_values = scatter_values
        self.linebreak_labels = linebreak_labels
        self.labels_max_len = labels_max_len
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
        
        self.validate_params()
        self.df_vals, self.df_ranks, self.pw_names = self.get_plot_data()
        #self.col_filt = self.get_col_filt()
        self.pw_colors, pw_color_is_dark = self.generate_large_palette(len(self.pw_names))
        self.str_index = self.df_vals.index.astype('str')

        self.bar_customdata = None
        self.bar_text = None
        self.insidetextfont = self.bar_label_font
        #self.inside_label_font_colors = self.get_inside_label_colors(pw_color_is_dark)
        self.inside_label_font = {**self.bar_label_font, "color": "#696969"}
        self.outside_label_font = {**self.bar_label_font, "color": "#696969"}#"#2a2a2a"}


    def get_extension(self):
        if self.data_filename:
            return self.data_filename.split('.')[-1]

    def get_bar_texttemplate(self, bar_texttemplate):
        if bar_texttemplate is None:
            if self.scatter_labels or self.plot_pws_yaxis:
                bar_texttemplate = '%{x:,.2f}'
            elif self.scatter_values:
                bar_texttemplate = '%{y}'
            else:
                bar_texttemplate = '%{y} %{x:.4s}'
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
        margin = {}

        left = margin.get("l", 80)
        right = margin.get("r", 80)
        top = margin.get("t", 100)
        bottom = margin.get("b", 80)

        plot_width = max(1, width - left - right)
        plot_height = max(1, height - top - bottom)
        return plot_width, plot_height

    def get_value_axis_bounds(self, bar_vals):
        axis_range = self.xlimit
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


    def labels_inside_bar(self, bar_vals):
        max_bar_val = max(bar_vals)
        return bar_vals >= max_bar_val / 2


    def validate_params(self):
        if isinstance(self.data_filename, str):
            if '.' not in self.data_filename:
                raise ValueError('`filename` must have an extension')
        elif self.data_filename is not None:
            raise TypeError('`filename` must be None or a string')



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
            return '%{y} - %{x:,.0f}<extra></extra>'
        return hovertemplate


    def create_pw_df_and_pw_names(self, pw_data_fpath):
        df = pd.read_csv(pw_data_fpath, index_col='global_index')
        df = df.drop('p.adj', axis=1)
        pw_names = df["pathway.name"].unique()
        with open(os.path.join(os.path.dirname(pw_data_fpath),'pathway_names_orig.txt'), 'w') as f:
                for pw in pw_names: f.write(pw+'\n')

        pw_idx_map = {name: i for i, name in enumerate(pw_names)}
        df["pathway.idx"] = df["pathway.name"].map(pw_idx_map)
        df = df.drop(columns=["pathway.name"])
        return df, pw_names


    def get_wide_df_and_lut(self):
        self.pw_data_fpath = './data/pathway_data.csv'
        if self.data_filename is None:
            self.data_filename = './data/pathway_data_wide.csv'
        if self.data_filename == './data/covid19.csv':
            df_wide = pd.read_csv(self.data_filename)
            return df_wide, None
        if os.path.exists(self.data_filename) and os.path.exists('./data/pathway_names.txt'):
            df_wide = pd.read_csv(self.data_filename, index_col='window')
            with open('./data/pathway_names.txt') as f:
                pw_names = np.array(f.readlines())
            self.mod_pw_data(pw_names)
        else: 
            df, pw_names = self.create_pw_df_and_pw_names(self.pw_data_fpath)
            df_wide = pd.pivot_table(df, values='-log10(p.adj)', index='window', columns='pathway.idx')
            df_wide = df_wide.fillna(0)
            df_wide.to_csv(self.data_filename)
            self.mod_pw_data(pw_names)
            with open(os.path.join(os.path.dirname(self.pw_data_fpath),'pathway_names_orig.txt'), 'w') as f:
                for pw in pw_names: f.write(pw+'\n')
        return df_wide, pw_names
    
    def get_orig_pw_names(self):
        with open(os.path.join(os.path.dirname(self.pw_data_fpath),'pathway_names_orig.txt')) as f:
            return np.array(f.readlines())

    def closest(self, lst, K):    
        return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

    def allOcc(self, s: str, ch):
        return [i for i, letter in enumerate(s) if letter == ch]

    def linebreak_pw_name(self, pw_name):
        occ = self.allOcc(pw_name, ' ')
        middle = np.round(len(pw_name)/2)
        pos = self.closest(occ, middle)
        return pw_name[:pos] + '<br>' + pw_name[pos+1:]

    def shorten_pw_name(self, pw_name):
        return pw_name[:self.labels_max_len-3] + '...' 

    def mod_pw_data(self, pw_names):
        if self.labels_max_len and len(pw_names[0])!=self.labels_max_len:
            pw_names = self.get_orig_pw_names() 
            for i, pw in enumerate(pw_names):
                pw_names[i] = self.shorten_pw_name(pw)
        if self.linebreak_labels and pw_names[0].find('<br>') == -1:
            pw_names = self.get_orig_pw_names()
            for i, pw in enumerate(pw_names):
                pw_names[i] = self.linebreak_pw_name(pw)
            

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
        df_wide_idx = range(df_wide_idx[-1]+1)
        df_wide = df_wide.reindex(df_wide_idx)

        df_wide_interp = df_wide.interpolate()

        topN = np.sort(df_wide_interp.to_numpy(), axis=1)[:, -self.n_bars:][:, ::-1]

        df_vals = pd.DataFrame(topN, index=df_wide_interp.index, columns=range(1, self.n_bars + 1))
        # get a dataframe with label (pathway) indices ranked after their value in each window
        df_ranks_wide = df_wide_interp.rank(axis=1, method='first', ascending=False)-1
        #df_ranks_wide = df_ranks_wide - (len(df_ranks_wide.columns)-self.n_bars)
        #print(df_ranks_wide.head(2))
        #df_ranks_wide[df_ranks_wide < 1] = np.nan
        df_ranks_wide[df_ranks_wide > self.n_bars-1] = np.nan
        ser = df_ranks_wide.stack().reset_index()

        df_ser = pd.DataFrame(ser).astype('int32')
        df_ranks = df_ser.pivot(index='window', columns=0, values=df_ser.columns[1])

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


    def get_frames(self):
        frames = []
        slider_steps = []

        bar_locs = np.arange(self.n_bars, 0, -1)

        if self.fixed_xaxis:
            self.val_ax_range = [0, self.df_vals.to_numpy().max() * 1.5]

        label_axis = dict(title_text = f'Top {self.n_bars} pathways', showticklabels= self.plot_pws_yaxis)

        if self.frame_subset is None: self.frame_subset = len(self.df_vals)
        for i in tqdm(range(len(self.df_vals[:self.frame_subset])), 'creating frames'):
            bar_vals_df = self.df_vals[self.df_vals.index==i]#self.df_vals.iloc[i, :self.n_bars].values

            data = self.get_data(i, bar_locs, bar_vals_df)

            if not(self.fixed_xaxis):
                self.val_ax_range = [0, max(bar_vals_df)* 1.5]

            value_axis = dict(showgrid=True, type=self.scale, title=self.val_ax_label, range=self.val_ax_range)#, tickformat=',.0f')
        
            frame_name = str(i)
            if self.slider and i % self.steps_per_period == 0:
                slider_steps.append({
                    "args": [[frame_name],
                            {"frame": {"duration": self.duration, "redraw": True},
                            "mode": "immediate",
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                            }],
                    "label": frame_name,
                    "method": "animate"
                })

            title_text = f'Locally enriched pathways per sliding window position {i/self.steps_per_period}'

            frame_layout = go.Layout(
                xaxis = value_axis,
                yaxis = label_axis, #annotations=self.get_annotations(i),
                autosize=False, width=1000, height=800,
                #margin = {'l':150, 'r':80, 't':100, 'b':120},
                title_text = title_text,
            )

            frames.append(go.Frame(data=data, layout=frame_layout, name=frame_name)) # name must be a string and slider steps must match the same string

        return frames, slider_steps
    

    def get_data(self, i, bar_locs, bar_vals_df):

        label_ids = self.df_ranks[self.df_ranks.index==i].values[0]#self.df_ranks.iloc[i].values
        label_ids_rev = np.flip(label_ids)
        x = bar_vals_df.transpose()[i].iloc[::-1]
        x.index = label_ids_rev
        label_names = np.flip(self.pw_names[label_ids])
        y = pd.Series(label_names, index = label_ids_rev)
        colors = self.pw_colors[label_ids_rev]

        if not(self.fixed_xaxis):
            self.val_ax_range = [0, max(x.values)*1.5]

        #bar_locs = bar_locs + np.random.rand(len(bar_locs)) / 10000 # done to prevent stacking of bars
        #x, y = (bar_vals, bar_locs)

        if not(self.scatter_labels): 
            val_labels = np.round(x, 2).astype(str)
            #self.bar_customdata = self.pw_names[label_ids]
            self.bar_text=np.char.add(self.pw_names[label_ids], np.char.add(['  '], val_labels))

        bar = go.Bar(
            x=x, y=y,
            ids=label_ids.astype(str),
            #customdata=self.bar_customdata,
            #text = self.bar_text,
            textposition=self.bar_textposition,
            hoverinfo='all',
            texttemplate=self.bar_texttemplate,
            textangle=0,            
            orientation='h',
            marker_color=colors,
            cliponaxis=False,
            #insidetextfont=self.insidetextfont,
            #outsidetextfont=self.bar_label_font,
            #hovertemplate=self.hovertemplate,
        )     

        if self.scatter_values:
            pass

        return [bar]


    
    # def get_annotations(self, i):        
    #     pos = i/self.steps_per_period
    #     period_label = {'xref': 'paper', 'yref': 'paper', 'font': {'size': 20},
    #                             'xanchor': 'right', 'showarrow': False, 'x' : .85, 'y': 1.1,
    #                             'text': 'pos.' + ' ' + str(pos)}
    #     return [period_label]
    

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
                          args=[None, {"frame": {"duration": self.duration, "redraw": True},#{"duration": 500, "redraw": False},#
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
        if self.out_filename:
            fig.write_html(self.out_filename, **self.write_html_kwargs)
        else:
            return fig