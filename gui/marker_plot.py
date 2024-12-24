import customtkinter as ctk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MarkerPlot:
    def __init__(self, parent):
        self.parent = parent
        
        # 데이터 관련 속성
        self.data = self.parent.data
        self.frame_idx = self.parent.frame_idx
        self.mouse_handler = self.parent.mouse_handler
        self.current_marker = None
        
        # 마커 표시 관련 속성
        self.show_names = self.parent.show_names
        self.pattern_selection_mode = self.parent.pattern_selection_mode
        self.pattern_markers = self.parent.pattern_markers
        
        # 플롯 관련 속성
        self.figure = None
        self.canvas = None
        self.axes = []
        self.lines = []
        
        # Sizer 초기화
        self.sizer = None
        self.sizer_dragging = False
        self.initial_sizer_x = None
        self.initial_panel_width = None
        
        # 필터 파라미터 초기화
        self.filter_params = {
            'butterworth': {
                'order': ctk.StringVar(value="4"),
                'cut_off_frequency': ctk.StringVar(value="10")
            },
            'kalman': {
                'trust_ratio': ctk.StringVar(value="20"),
                'smooth': ctk.StringVar(value="1")
            },
            'gaussian': {
                'sigma_kernel': ctk.StringVar(value="3")
            },
            'LOESS': {
                'nb_values_used': ctk.StringVar(value="10")
            },
            'median': {
                'kernel_size': ctk.StringVar(value="3")
            }
        }
        self.filter_type_var = ctk.StringVar(value='butterworth')
        
        # GUI 요소 초기화 - 반드시 parent의 GUI 요소가 모두 초기화된 후에 실행
        self.right_panel = ctk.CTkFrame(self.parent, fg_color="black")
        self.graph_frame = ctk.CTkFrame(self.right_panel, fg_color="black")
        self.main_content = self.parent.main_content
        
        # 편집 관련 속성
        self.editing = False
        
    def update_data(self):
        """부모 클래스의 데이터를 업데이트합니다."""
        self.data = self.parent.data
        self.frame_idx = self.parent.frame_idx
        self.show_names = self.parent.show_names
        self.pattern_selection_mode = self.parent.pattern_selection_mode
        self.pattern_markers = self.parent.pattern_markers

    def show_marker_plot(self, marker_name):
        # 데이터 업데이트
        self.update_data()
        
        # 데이터 유효성 체크
        if self.data is None or len(self.data) == 0:
            print("No data available")
            return
            
        # Save current states
        was_editing = getattr(self, 'editing', False)
        
        # Save previous filter parameters if they exist
        prev_filter_params = None
        if hasattr(self, 'filter_params'):
            prev_filter_params = {
                filter_type: {
                    param: var.get() for param, var in params.items()
                } for filter_type, params in self.filter_params.items()
            }
        prev_filter_type = getattr(self, 'filter_type_var', None)
        if prev_filter_type:
            prev_filter_type = prev_filter_type.get()

        if not self.graph_frame.winfo_ismapped():
            # Right panel 표시
            self.right_panel.pack(side='right', fill='both')
            
            # 초기 너비 설정 (전체 창의 1/3)
            initial_width = self.parent.winfo_width() // 3
            self.right_panel.configure(width=initial_width)
            
            # Sizer 생성 및 설정
            if not hasattr(self, 'sizer') or self.sizer is None:
                self.sizer = ctk.CTkFrame(self.main_content, width=5, height=self.main_content.winfo_height(),
                                        fg_color="#666666", bg_color="black")
                self.sizer.pack(side='left', fill='y')
                self.sizer.pack_propagate(False)
                
                # Sizer bindings
                self.sizer.bind('<Enter>', lambda e: (
                    self.sizer.configure(fg_color="#888888"),
                    self.sizer.configure(cursor="sb_h_double_arrow")
                ))
                self.sizer.bind('<Leave>', lambda e: self.sizer.configure(fg_color="#666666"))
                self.sizer.bind('<Button-1>', self.start_resize)
                self.sizer.bind('<B1-Motion>', self.do_resize)
                self.sizer.bind('<ButtonRelease-1>', self.stop_resize)
        
            self.graph_frame.pack(fill='both', expand=True)

        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        self.marker_plot_fig = Figure(figsize=(6, 8), facecolor='black')
        self.marker_plot_fig.patch.set_facecolor('black')

        self.current_marker = marker_name

        self.marker_axes = []
        self.marker_lines = []
        coords = ['X', 'Y', 'Z']

        if not hasattr(self, 'outliers') or marker_name not in self.outliers:
            self.outliers = {marker_name: np.zeros(len(self.data), dtype=bool)}

        outlier_frames = np.where(self.outliers[marker_name])[0]

        for i, coord in enumerate(coords):
            ax = self.marker_plot_fig.add_subplot(3, 1, i+1)
            ax.set_facecolor('black')

            data = self.data[f'{marker_name}_{coord}']
            frames = np.arange(len(data))

            ax.plot(frames[~self.outliers[marker_name]],
                    data[~self.outliers[marker_name]],
                    color='white',
                    label='Normal')

            if len(outlier_frames) > 0:
                ax.plot(frames[self.outliers[marker_name]],
                        data[self.outliers[marker_name]],
                        'ro',
                        markersize=3,
                        label='Outlier')

            ax.set_title(f'{marker_name} - {coord}', color='white')
            ax.grid(True, color='gray', alpha=0.3)
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color('white')

            self.marker_axes.append(ax)

            if len(outlier_frames) > 0:
                ax.legend(facecolor='black',
                        labelcolor='white',
                        loc='upper right',
                        bbox_to_anchor=(1.0, 1.0))

        # initialize current frame display line
        self.marker_lines = []  # initialize existing lines
        for ax in self.marker_axes:
            line = ax.axvline(x=self.frame_idx, color='red', linestyle='--', alpha=0.8)
            self.marker_lines.append(line)

        self.marker_plot_fig.tight_layout()

        self.marker_canvas = FigureCanvasTkAgg(self.marker_plot_fig, master=self.graph_frame)
        self.marker_canvas.draw()
        self.marker_canvas.get_tk_widget().pack(fill='both', expand=True)

        self.initial_graph_limits = []
        for ax in self.marker_axes:
            self.initial_graph_limits.append({
                'x': ax.get_xlim(),
                'y': ax.get_ylim()
            })

        self.connect_mouse_events()

        button_frame = ctk.CTkFrame(self.graph_frame)
        button_frame.pack(fill='x', padx=5, pady=5)

        reset_button = ctk.CTkButton(button_frame,
                                    text="Reset View",
                                    command=self.reset_graph_view,
                                    width=80,
                                    fg_color="#333333",
                                    hover_color="#444444")
        reset_button.pack(side='right', padx=5)

        # Edit button to open the new window
        self.edit_button = ctk.CTkButton(button_frame,
                                        text="Edit",
                                        command=self.toggle_edit_window,  # window rather than menu
                                        width=80,
                                        fg_color="#333333",
                                        hover_color="#444444")
        self.edit_button.pack(side='right', padx=5)

        # Restore previous parameter values if they exist
        if prev_filter_params:
            for filter_type, params in prev_filter_params.items():
                for param, value in params.items():
                    self.filter_params[filter_type][param].set(value)

        # Backwards compatibility for filter parameters
        self.hz_var = self.filter_params['butterworth']['cut_off_frequency']
        self.filter_order_var = self.filter_params['butterworth']['order']

        self.selection_data = {
            'start': None,
            'end': None,
            'rects': [],
            'current_ax': None,
            'rect': None
        }

        # Restore edit state if it was active
        if was_editing:
            self.start_edit()

        # initialize selection_data
        self.selection_data = {
            'start': None,
            'end': None,
            'rects': []
        }

        # initialize selection_in_progress
        self.selection_in_progress = False

        # update marker name display logic
        if self.show_names or (hasattr(self, 'pattern_selection_mode') and self.pattern_selection_mode):
            for marker in self.marker_names:
                x = self.data.loc[self.frame_idx, f'{marker}_X']
                y = self.data.loc[self.frame_idx, f'{marker}_Y']
                z = self.data.loc[self.frame_idx, f'{marker}_Z']
                
                # determine marker name color
                if hasattr(self, 'pattern_selection_mode') and self.pattern_selection_mode and marker in self.pattern_markers:
                    name_color = 'red'  # pattern-based selected marker
                else:
                    name_color = 'black'  # normal marker
                    
                if not np.isnan(x) and not np.isnan(y) and not np.isnan(z):
                    if self.is_z_up:
                        self.ax.text(x, y, z, marker, color=name_color)
                    else:
                        self.ax.text(x, z, y, marker, color=name_color)
                        
    def start_resize(self, event):
        self.sizer_dragging = True
        self.initial_sizer_x = event.x_root
        self.initial_panel_width = self.right_panel.winfo_width()

    def do_resize(self, event):
        if self.sizer_dragging:
            dx = event.x_root - self.initial_sizer_x
            new_width = max(200, min(self.initial_panel_width - dx, self.parent.winfo_width() - 200))
            self.right_panel.configure(width=new_width)

    def stop_resize(self, event):
        self.sizer_dragging = False
    
    def connect_mouse_events(self):
        """마커 캔버스의 마우스 이벤트를 연결합니다."""
        if hasattr(self, 'marker_canvas') and self.marker_canvas:
            self.marker_canvas.mpl_connect('scroll_event', self.mouse_handler.on_marker_scroll)
            self.marker_canvas.mpl_connect('button_press_event', self.mouse_handler.on_marker_mouse_press)
            self.marker_canvas.mpl_connect('button_release_event', self.mouse_handler.on_marker_mouse_release)
            self.marker_canvas.mpl_connect('motion_notify_event', self.mouse_handler.on_marker_mouse_move)
    
    def reset_graph_view(self):
        if hasattr(self, 'marker_axes') and hasattr(self, 'initial_graph_limits'):
            for ax, limits in zip(self.marker_axes, self.initial_graph_limits):
                ax.set_xlim(limits['x'])
                ax.set_ylim(limits['y'])
            self.marker_canvas.draw()

    def toggle_edit_window(self):
        try:
            # focus on existing edit_window if it exists
            if hasattr(self, 'edit_window') and self.edit_window:
                self.edit_window.focus()
            else:
                # create new EditWindow
                self.edit_window = EditWindow(self)
                self.edit_window.focus()
                
        except Exception as e:
            print(f"Error in toggle_edit_window: {e}")
            import traceback
            traceback.print_exc()