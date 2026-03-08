import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from tqdm import tqdm

from .utils import color_generation



class _BarChartRace:
    
    def __init__(self, data_filename=None, data_is_wide=False, out_filename=None, n_bars=None, 
                          fixed_order=False, fixed_max=False, steps_per_period=10, 
                          period_length=500, end_period_pause=0, interpolate_period=True, 
                          perpendicular_bar_func=None, title=None, bar_size=.95, 
                          bar_textposition='outside', bar_texttemplate=None, bar_label_font=None, 
                          tick_label_font=None, hovertemplate=None, slider=True, scale='linear', 
                          write_html_kwargs=None,
                          fixed_xaxis = False, plot_pws_yaxis=False, val_ax_label=None, 
                          scatter_labels=False, scatter_values_inside_bar=False, linebreak_labels=False, 
                          labels_max_len=None, linebreak_labels_len_greater=None, 
                          plot_labels_over_bars=False, bargap=None, yaxis_title_standoff=30, 
                          bar_switching_anim=False, frame_subset=None):
        
        self.data_filename = data_filename
        self.data_is_wide = data_is_wide
        self.out_filename = out_filename
        
        self.fixed_xaxis = fixed_xaxis
        self.plot_pws_yaxis = plot_pws_yaxis
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
        self.plot_labels_over_bars = plot_labels_over_bars
        self.bargap = self.get_bargap(bargap)
        
        self.yaxis_title_standoff = yaxis_title_standoff
        self.scatter_values_inside_bar = scatter_values_inside_bar
        self.plot_labels_over_bars = plot_labels_over_bars
        self.bargap = self.get_bargap(bargap)
        
        self.yaxis_title_standoff = yaxis_title_standoff
        self.scatter_values_inside_bar = scatter_values_inside_bar
        self.linebreak_labels = linebreak_labels
        self.labels_max_len = labels_max_len
        self.linebreak_labels_len_greater = linebreak_labels_len_greater

        self.linebreak_labels_len_greater = linebreak_labels_len_greater

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
        self.pw_colors, pw_color_is_dark = color_generation.generate_large_palette(len(self.pw_names))
        self.str_index = self.df_vals.index.astype('str')

        self.bar_customdata = None
        self.bar_text = None
        self.insidetextfont = self.bar_label_font
        self.inside_label_font_colors = np.array(["#ffffff" if val else "#2a2a2a" for val in pw_color_is_dark])
        self.inside_label_font_colors = np.array(["#ffffff" if val else "#2a2a2a" for val in pw_color_is_dark])
        self.outside_label_font = {**self.bar_label_font, "color": "#696969"}#"#2a2a2a"}

        self.bar_switching_anim = bar_switching_anim
        self.slider_step_dict, self.play_args = self.get_animation_opts()

    def get_animation_opts(self):
        slider_step_dict = {"frame": {"duration": self.duration, "redraw": False},
                            "mode": "immediate"}
        play_args = [None, {"frame": {"duration": self.duration},
                            "fromcurrent": True}]
        if self.bar_switching_anim:
            self.redraw = True
            slider_step_dict["fromcurrent"] = True,
            slider_step_dict["transition"] = {"duration": self.duration}
    
        else:
            self.redraw = False
            play_args[1]["transition"] = {"duration": self.duration,
                                        "easing": "linear"}
        play_args[1]["frame"]["redraw"] = self.redraw
        return slider_step_dict, play_args


    def get_label_lens(self):
        if not(self.linebreak_labels or self.linebreak_labels_len_greater is not None):
            label_lens = np.array([len(pw)/4 for pw in self.pw_names])
        else:
            label_lens = []
            for pw in self.pw_names:
                if self.linebreak_labels or self.linebreak_labels_len_greater is not None:
                    label_lens.append(max([len(part)/7 for part in pw.split('<br>')]))
            label_lens = np.int32(label_lens)
        self.label_lens = label_lens + 1


    def get_glob_max_needed_xaxis_len(self):
        label_lens_of_bars = self.label_lens[self.df_ranks.to_numpy().T]
        bar_lens = self.df_vals.to_numpy().T

        if self.plot_labels_over_bars:
            max_bar_len = bar_lens.max()
            max_label_len = label_lens_of_bars.max()
            if not(self.scatter_values_inside_bar) and self.bar_textposition == 'outside':
                max_bar_len_idx = bar_lens.argmax()
                max_bar_len += bar_lens.flatten()[max_bar_len_idx]/4 + 1 + 0.5  # 0.5 for float formatting with two decimal places
                                                                                # needs to be handled better!
                total_lens = max(max_bar_len, max_label_len)
        else:    
            total_lens = bar_lens + label_lens_of_bars
            max_len = total_lens.max()
            if not(self.scatter_values_inside_bar) and self.bar_textposition == 'outside':
                max_idx = total_lens.argmax() 
                max_len += total_lens.flatten()[max_idx]/4 + 1

        return max_len

    def get_bargap(self, bargap):
        if bargap is None:
            bargap = 0.35 if self.plot_labels_over_bars else 0.15
        return bargap

    def get_extension(self):
        if self.data_filename:
            return self.data_filename.split('.')[-1]

    def get_bar_texttemplate(self, bar_texttemplate):
        if bar_texttemplate is None:
            if self.scatter_labels or self.plot_pws_yaxis or self.plot_labels_over_bars:
                self.scatter_values_inside_bar = False
            if self.scatter_labels or self.plot_pws_yaxis or self.plot_labels_over_bars:
                self.scatter_values_inside_bar = False
                bar_texttemplate = '%{x:,.2f}'
            elif self.scatter_values_inside_bar:
                bar_texttemplate = '%{y}'
            else:
                bar_texttemplate = '%{y} %{x:.2f}'
                bar_texttemplate = '%{y} %{x:.2f}'
        return bar_texttemplate


    def get_scatter_texttemplate(self, scatter_texttemplate):
        if scatter_texttemplate is None:
            scatter_texttemplate = "%{customdata}"
        return scatter_texttemplate
    
    

    def validate_params(self):
        if isinstance(self.data_filename, str):
            if '.' not in self.data_filename:
                raise ValueError('`filename` must have an extension')
        elif self.data_filename is not None:
            raise TypeError('`filename` must be None or a string')


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
        data_dir = './data'
        data_dir = './data'
        if self.data_filename is None:
            self.data_filename = 'pathway_data_wide.csv'
        self.data_fpath = os.path.join(data_dir,self.data_filename)

        if self.data_filename == 'covid19.csv':
            df_wide = pd.read_csv(self.data_fpath, index_col='date')
            df_wide = df_wide.fillna(0)
            df_wide.index = range(len(df_wide.index))
            countries = df_wide.columns
            df_wide.columns = [i for i in range(len(countries))]
            return df_wide, countries
        
        if self.data_filename == 'FAOSTAT_data.csv':
            df = pd.read_csv(self.data_fpath, index_col='Year')
            df.sort_index(inplace=True)
            items = df['Item'].unique()
            item_idx_map = {name: i for i, name in enumerate(items)}
            df["Item Index"] = df["Item"].map(item_idx_map)

            df.drop(df.columns.difference(['Item Index','Value']), axis=1, inplace=True)
            df_wide = pd.pivot_table(df, values='Value', index='Year', columns='Item Index')
            df_wide = df_wide.fillna(0)
            df_wide.index = range(len(df_wide.index))
            return df_wide, items
        
        if os.path.exists(self.data_fpath) and os.path.exists('./data/pathway_names.txt'):
            df_wide = pd.read_csv(self.data_fpath, index_col='window')
            self.data_filename = 'pathway_data_wide.csv'
        self.data_fpath = os.path.join(data_dir,self.data_filename)

        if self.data_filename == 'covid19.csv':
            df_wide = pd.read_csv(self.data_fpath, index_col='date')
            df_wide = df_wide.fillna(0)
            df_wide.index = range(len(df_wide.index))
            countries = df_wide.columns
            df_wide.columns = [i for i in range(len(countries))]
            return df_wide, countries
        
        if self.data_filename == 'FAOSTAT_data.csv':
            df = pd.read_csv(self.data_fpath, index_col='Year')
            df.sort_index(inplace=True)
            items = df['Item'].unique()
            item_idx_map = {name: i for i, name in enumerate(items)}
            df["Item Index"] = df["Item"].map(item_idx_map)

            df.drop(df.columns.difference(['Item Index','Value']), axis=1, inplace=True)
            df_wide = pd.pivot_table(df, values='Value', index='Year', columns='Item Index')
            df_wide = df_wide.fillna(0)
            df_wide.index = range(len(df_wide.index))
            return df_wide, items
        
        if os.path.exists(self.data_fpath) and os.path.exists('./data/pathway_names.txt'):
            df_wide = pd.read_csv(self.data_fpath, index_col='window')
            with open('./data/pathway_names.txt') as f:
                pw_names = np.array(f.readlines())
            self.mod_pw_data(pw_names)
        else: 
            self.pw_data_fpath = './data/pathway_data.csv'
            self.pw_data_fpath = './data/pathway_data.csv'
            df, pw_names = self.create_pw_df_and_pw_names(self.pw_data_fpath)
            with open(os.path.join(os.path.dirname(self.pw_data_fpath),'pathway_names_orig.txt'), 'w') as f:
                for pw in pw_names: f.write(pw+'\n')

            with open(os.path.join(os.path.dirname(self.pw_data_fpath),'pathway_names_orig.txt'), 'w') as f:
                for pw in pw_names: f.write(pw+'\n')

            df_wide = pd.pivot_table(df, values='-log10(p.adj)', index='window', columns='pathway.idx')
            df_wide = df_wide.fillna(0)
            df_wide.to_csv(self.data_filename)
            self.mod_pw_data(pw_names)
    
    
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
        if self.labels_max_len is not None and len(pw_names[0])!=self.labels_max_len:
            pw_names = self.get_orig_pw_names() 
            for i, pw in enumerate(pw_names):
                pw_names[i] = self.shorten_pw_name(pw)
        if self.linebreak_labels is not None and pw_names[0].find('<br>') == -1:

            pw_names = self.get_orig_pw_names()
            for i, pw in enumerate(pw_names):
                pw_names[i] = self.linebreak_pw_name(pw)
        elif self.linebreak_labels_len_greater is not None: 
            pw_names = self.get_orig_pw_names()
            for i, pw in enumerate(pw_names):
                if len(pw) > self.linebreak_labels_len_greater:
                    pw_names[i] = self.linebreak_pw_name(pw)
        return pw_names    
    

    def get_plot_data(self):
        df_wide, pw_names = self.get_wide_df_and_lut()
        if self.n_bars is None:
            self.n_bars = len(df_wide.iloc[0].values)
            self.n_bars = len(df_wide.iloc[0].values)
        else: 
            self.n_bars = min(self.n_bars, len(df_wide.iloc[0].values))
            self.n_bars = min(self.n_bars, len(df_wide.iloc[0].values))

        df_wide_idx = df_wide.index
        if df_wide.index[0] == 1:
            df_wide_idx -= 1

        df_wide.index = df_wide_idx * self.steps_per_period
        df_wide_idx = range(df_wide.index[-1]+1)
        df_wide = df_wide.reindex(df_wide_idx)

        df_wide_interp = df_wide.interpolate()
        df_wide_interp.to_csv('df_wide_interp.csv')
        # sort dataframe asc as matrix, grab n largest columns and reverse them
        top_n = np.sort(df_wide_interp.to_numpy(), axis=1)[:, -self.n_bars:][:, ::-1]

        df_vals = pd.DataFrame(top_n, index=df_wide_interp.index, columns=range(1, self.n_bars + 1))

        # get a dataframe with label (pathway) indices ranked after their value in each window
        df_ranks_wide = df_wide_interp.rank(axis=1, method='first', ascending=False)-1


        df_ranks_wide[df_ranks_wide > self.n_bars-1] = np.nan
        ser = df_ranks_wide.stack().reset_index()

        df_ser = pd.DataFrame(ser).astype('int32')
        df_ranks = df_ser.pivot(index=df_ser.columns[0], columns=0, values=df_ser.columns[1])
        df_ranks = df_ser.pivot(index=df_ser.columns[0], columns=0, values=df_ser.columns[1])

        return df_vals, df_ranks, pw_names


    def get_frames(self):
        frames = []
        slider_steps = []

        self.y_coords = np.arange(self.n_bars)
        self.y_coords = np.arange(self.n_bars)

        self.get_label_lens()
        self.get_label_lens()
        if self.fixed_xaxis:
            self.val_ax_range = [0, self.get_glob_max_needed_xaxis_len()]
            self.val_ax_range = [0, self.get_glob_max_needed_xaxis_len()]


        if self.frame_subset is None: self.frame_subset = len(self.df_vals)
        for i in tqdm(range(len(self.df_vals[:self.frame_subset])), 'creating frames'):
            bar_vals = self.df_vals.iloc[i, :self.n_bars].values
            bar_vals = self.df_vals.iloc[i, :self.n_bars].values

            data, annotations, current_labels = self.get_data(i, bar_vals)

            value_axis = dict(showgrid=True, type=self.scale, title=self.val_ax_label, range=self.val_ax_range)
        
            label_axis = dict(
                title_text = f'Top {self.n_bars} pathways', 
                title_standoff=self.yaxis_title_standoff,
                showticklabels=self.plot_pws_yaxis,
                tickmode='array',
                tickvals=np.arange(self.n_bars),
                ticktext=current_labels,
                range=[-self.bar_size / 2, self.n_bars - 1 + self.bar_size / 2 + self.bargap]
            )
            
            frame_name = str(i)
            if self.slider and i % self.steps_per_period == 0:
                slider_steps.append({
                    "args": [[frame_name],
                            self.slider_step_dict],
                    "label": frame_name,
                    "method": "animate"
                })

            title_text = f'Locally enriched pathways per sliding window position'
            title = {'text': title_text,
                     'font': {'size': 20},
                    'x': 0.5, 'xref': 'paper',
                    'xanchor': 'center', 'yanchor': 'top'}
            
            title_text = f'Locally enriched pathways per sliding window position'
            title = {'text': title_text,
                     'font': {'size': 20},
                    'x': 0.5, 'xref': 'paper',
                    'xanchor': 'center', 'yanchor': 'top'}
            
            frame_layout = go.Layout(
                xaxis = value_axis,
                yaxis = label_axis,
                annotations = annotations,
                autosize=False, width=1000, height=800+self.n_bars*10,
                title = title,
                bargap = self.bargap,
                yaxis = label_axis,
                annotations = annotations,
                autosize=False, width=1000, height=800+self.n_bars*10,
                title = title,
                bargap = self.bargap,
            )

            frames.append(go.Frame(data=data, layout=frame_layout, name=frame_name))
            frames.append(go.Frame(data=data, layout=frame_layout, name=frame_name))

        return frames, slider_steps
    

    def get_data(self, i, bar_vals):

        label_ids = self.df_ranks[self.df_ranks.index==i].values[0]
        bar_vals_rev = np.flip(bar_vals)
        label_ids_rev = np.flip(label_ids)
        label_names = None

        if self.bar_switching_anim:
            x = bar_vals_rev
            x[x==0] = np.nan
        else:
            x = pd.Series(bar_vals_rev, index=label_ids_rev)
            x.replace(0, np.nan, inplace=True)
            #y = pd.Series(label_names, index = label_ids_rev)
        
        label_names = np.flip(self.pw_names[label_ids])          

        colors = self.pw_colors[label_ids_rev]

        if not(self.fixed_xaxis):
            self.val_ax_range = [0, max(bar_vals_rev+self.label_lens[label_ids_rev])]

        annotations = []
        if not self.plot_pws_yaxis:
            for j in range(1, self.n_bars + 1):
                if j % 5 == 0:
                    annotations.append(dict(
                        x=0, y=j-1,
                        xref="paper", yref="y",
                        text=str(j), showarrow=False,
                        xanchor="right", align="right"
                    ))

        #print('x: ', x)
        #print('y: ', self.y_coords)

        bar = go.Bar(
            x=x, 
            y=self.y_coords,
            textposition=self.bar_textposition,
            hoverinfo='all',
            texttemplate=self.bar_texttemplate,
            textangle=0,            
            orientation='h',
            marker_color=colors,
            cliponaxis=False,
            showlegend=False,
        )    

        if self.scatter_values_inside_bar:
            inside_label_font = {"color": self.inside_label_font_colors[label_ids_rev]}

            scatter = go.Scatter(
                x=x, 
                y=self.y_coords,
                mode="text",
                texttemplate="%{x:,.2f}  ",
                textposition="middle left",
                textfont=inside_label_font,
                cliponaxis=False,
                hoverinfo="skip",
                showlegend=False
            )
            return [bar, scatter], annotations, label_names

        if self.plot_labels_over_bars:
            x_ = np.zeros(self.n_bars)
            y_ = self.y_coords + 0.45
            scatter = go.Scatter(
                x=x_, y=y_,
                customdata=label_names,
                mode="text",
                texttemplate="%{customdata}",
                textposition="middle right",
                cliponaxis=False,
                hoverinfo="skip",
                showlegend=False
            )
            return [bar, scatter], annotations, label_names

        return [bar], annotations, label_names

    
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
                          args= self.play_args),
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
                            "font": {"size": 20},
                            "prefix": "Position: ",
                            "visible": True,
                            "xanchor": "right"
                        },
                        #"fromcurrent": True,
                        "transition": {"duration": self.duration, "easing": "cubic-in-out"},#transition duration must be set at least as long as frame duration
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

