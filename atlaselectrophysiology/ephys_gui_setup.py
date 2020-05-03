from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
from pyqtgraph.widgets import MatplotlibWidget as matplot
import numpy as np
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')


class Setup():
    def init_layout(self, main_window):
        self.resize(1600, 800)
        self.setWindowTitle('Electrophysiology Atlas')
        self.setSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QGridLayout()

        self.init_menubar()
        self.init_interaction_features()
        self.init_figures()

        main_layout = QtWidgets.QGridLayout()
        main_layout.addWidget(self.fig_data_area, 0, 0, 10, 1)
        main_layout.addWidget(self.fig_hist_area, 0, 1, 10, 1)
        main_layout.addLayout(self.interaction_layout3, 0, 2, 1, 1)
        main_layout.addWidget(self.fig_slice, 1, 2, 3, 1)
        main_layout.addLayout(self.interaction_layout1, 4, 2, 1, 1)
        main_layout.addWidget(self.fig_fit, 5, 2, 3, 1)
        main_layout.addLayout(self.interaction_layout2, 8, 2, 2, 1)
        main_layout.setColumnStretch(0, 6)
        main_layout.setColumnStretch(1, 2)
        main_layout.setColumnStretch(2, 2)

        main_widget.setLayout(main_layout)

    def init_menubar(self):
        """
        Create menu bar and add all possible plot and keyboard interaction options
        """
        # Create menubar widget and add it to the main GUI window
        menu_bar = QtWidgets.QMenuBar(self)
        self.setMenuBar(menu_bar)

        # Define all 2D scatter/ image plot options
        scatter_drift = QtGui.QAction('Drift Map Scatter', self)
        scatter_drift.triggered.connect(lambda: self.plot_scatter(self.scat_drift_data))
        img_drift = QtGui.QAction('Drift Map Image', self)
        img_drift.triggered.connect(lambda: self.plot_image(self.img_drift_data))
        img_corr = QtGui.QAction('Correlation', self)
        img_corr.triggered.connect(lambda: self.plot_image(self.img_corr_data))
        img_rmsAP = QtGui.QAction('rms AP', self)
        img_rmsAP.triggered.connect(lambda: self.plot_image(self.img_rms_APdata))
        img_rmsLFP = QtGui.QAction('rms LFP', self)
        img_rmsLFP.triggered.connect(lambda: self.plot_image(self.img_rms_LFdata))
        # Add menu bar for 2D scatter/ image plot options
        img_options = menu_bar.addMenu('Image Plots')
        img_options.addAction(scatter_drift)
        img_options.addAction(img_drift)
        img_options.addAction(img_corr)
        img_options.addAction(img_rmsAP)
        img_options.addAction(img_rmsLFP)

        # Define all 1D line plot options
        line_fr = QtGui.QAction('Firing Rate', self)
        line_fr.triggered.connect(lambda: self.plot_line(self.line_fr_data))
        line_amp = QtGui.QAction('Amplitude', self)
        line_amp.triggered.connect(lambda: self.plot_line(self.line_amp_data))
        # Add menu bar for 1D line plot options
        line_options = menu_bar.addMenu('Line Plots')
        line_options.addAction(line_fr)
        line_options.addAction(line_amp)

        # Define all 2D probe plot options
        probe_options = menu_bar.addMenu("Probe Plots")
        probe_rmsAP = QtGui.QAction('rms AP', self)
        probe_rmsAP.triggered.connect(lambda: self.plot_probe(self.probe_rms_APdata))
        probe_rmsLFP = QtGui.QAction('rms LF', self)
        probe_rmsLFP.triggered.connect(lambda: self.plot_probe(self.probe_rms_LFdata))
        # Add menu bar for 2D probe plot options
        probe_options.addAction(probe_rmsAP)
        probe_options.addAction(probe_rmsLFP)

        # Add the different frequency band options in a loop. These must be the same as in
        # load_data
        freq_bands = np.vstack(([0, 4], [4, 10], [10, 30], [30, 80], [80, 200]))
        for iF, freq in enumerate(freq_bands):
            band = f"{freq[0]} - {freq[1]} Hz"
            probe = QtGui.QAction(band, self)
            probe.triggered.connect(lambda checked, item=band: self.plot_probe(
                                    self.probe_lfp_data[item]))
            probe_options.addAction(probe)


        # Define all possible keyboard interactions for GUI
        # Shortcut to apply interpolation
        fit_option = QtGui.QAction('Fit', self)
        fit_option.setShortcut('Return')
        # Shortcuts to apply offset
        fit_option.triggered.connect(self.fit_button_pressed)
        offset_option = QtGui.QAction('Offset', self)
        offset_option.setShortcut('O')
        offset_option.triggered.connect(self.offset_button_pressed)
        moveup_option = QtGui.QAction('Offset + 50um', self)
        moveup_option.setShortcut('Shift+Up')
        moveup_option.triggered.connect(self.moveup_button_pressed)
        movedown_option = QtGui.QAction('Offset - 50um', self)
        movedown_option.setShortcut('Shift+Down')
        movedown_option.triggered.connect(self.movedown_button_pressed)
        # Shortcut to hide/show region labels
        toggle_labels_option = QtGui.QAction('Hide/Show Labels', self)
        toggle_labels_option.setShortcut('Shift+A')
        toggle_labels_option.triggered.connect(self.toggle_labels_button_pressed)
        # Shortcut to hide/show reference lines
        toggle_lines_option = QtGui.QAction('Hide/Show Lines', self)
        toggle_lines_option.setShortcut('Shift+L')
        toggle_lines_option.triggered.connect(self.toggle_line_button_pressed)
        # Shortcut to remove a reference line
        delete_line_option = QtGui.QAction('Remove Line', self)
        delete_line_option.setShortcut('Del')
        delete_line_option.triggered.connect(self.delete_line_button_pressed)
        # Shortcut to move between previous/next moves
        next_option = QtGui.QAction('Next', self)
        next_option.setShortcut('Right')
        next_option.triggered.connect(self.next_button_pressed)
        prev_option = QtGui.QAction('Previous', self)
        prev_option.setShortcut('Left')
        prev_option.triggered.connect(self.prev_button_pressed)
        # Shortcut to reset GUI to initial state
        reset_option = QtGui.QAction('Reset', self)
        reset_option.setShortcut('Shift+R')
        # Shortcut to upload final state to Alyx
        reset_option.triggered.connect(self.reset_button_pressed)
        complete_option = QtGui.QAction('Complete', self)
        complete_option.setShortcut('Shift+F')
        complete_option.triggered.connect(self.complete_button_pressed)

        # Shortcuts to switch between different views on left most data plot
        view1_option = QtGui.QAction('View 1', self)
        view1_option.setShortcut('Shift+1')
        view1_option.triggered.connect(lambda: self.set_view(view=1))
        view2_option = QtGui.QAction('View 2', self)
        view2_option.setShortcut('Shift+2')
        view2_option.triggered.connect(lambda: self.set_view(view=2))
        view3_option = QtGui.QAction('View 3', self)
        view3_option.setShortcut('Shift+3')
        view3_option.triggered.connect(lambda: self.set_view(view=3))

        # Add menu bar with all possible keyboard interactions
        shortcut_options = menu_bar.addMenu("Shortcut Keys")
        shortcut_options.addAction(fit_option)
        shortcut_options.addAction(offset_option)
        shortcut_options.addAction(moveup_option)
        shortcut_options.addAction(movedown_option)
        shortcut_options.addAction(toggle_labels_option)
        shortcut_options.addAction(toggle_lines_option)
        shortcut_options.addAction(delete_line_option)
        shortcut_options.addAction(next_option)
        shortcut_options.addAction(prev_option)
        shortcut_options.addAction(reset_option)
        shortcut_options.addAction(complete_option)
        shortcut_options.addAction(view1_option)
        shortcut_options.addAction(view2_option)
        shortcut_options.addAction(view3_option)

    def init_interaction_features(self):
        """
        Create all interaction widgets that will be added to the GUI
        """
        # Button to apply interpolation
        self.fit_button = QtWidgets.QPushButton('Fit', font=self.font)
        self.fit_button.clicked.connect(self.fit_button_pressed)
        # Button to apply offset
        self.offset_button = QtWidgets.QPushButton('Offset', font=self.font)
        self.offset_button.clicked.connect(self.offset_button_pressed)
        # Button to go to next move
        self.next_button = QtWidgets.QPushButton('Next', font=self.font)
        self.next_button.clicked.connect(self.next_button_pressed)
        # Button to go to previous move
        self.prev_button = QtWidgets.QPushButton('Previous', font=self.font)
        self.prev_button.clicked.connect(self.prev_button_pressed)
        # String to display current move index
        self.idx_string = QtWidgets.QLabel(font=self.font)
        # String to display total number of moves
        self.tot_idx_string = QtWidgets.QLabel(font=self.font)
        # Button to reset GUI to initial state
        self.reset_button = QtWidgets.QPushButton('Reset', font=self.font)
        self.reset_button.clicked.connect(self.reset_button_pressed)
        # Button to upload final state to Alyx
        self.complete_button = QtWidgets.QPushButton('Complete', font=self.font)
        self.complete_button.clicked.connect(self.complete_button_pressed)
        # Drop down list to choose subject
        self.subj_list = QtGui.QStandardItemModel()
        subj_combobox = QtWidgets.QComboBox()
        # Add line edit and completer to be able to search for subject
        subj_combobox.setLineEdit(QtWidgets.QLineEdit())
        subj_completer = QtWidgets.QCompleter()
        subj_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        subj_combobox.setCompleter(subj_completer)
        subj_combobox.setModel(self.subj_list)
        subj_combobox.completer().setModel(self.subj_list)
        subj_combobox.textActivated.connect(self.on_subject_selected)
        # Drop down list to choose session
        self.sess_list = QtGui.QStandardItemModel()
        sess_combobox = QtWidgets.QComboBox()
        sess_combobox.setModel(self.sess_list)
        sess_combobox.activated.connect(self.on_session_selected)
        # Button to get data to display in GUI
        self.data_button = QtWidgets.QPushButton('Get Data', font=self.font)
        self.data_button.clicked.connect(self.data_button_pressed)

        # Arrange interaction features into three different layout groups
        # Group 1
        hlayout1 = QtWidgets.QHBoxLayout()
        hlayout2 = QtWidgets.QHBoxLayout()
        hlayout1.addWidget(self.fit_button, stretch=1)
        hlayout1.addWidget(self.offset_button, stretch=1)
        hlayout1.addWidget(self.tot_idx_string, stretch=2)
        hlayout2.addWidget(self.prev_button, stretch=1)
        hlayout2.addWidget(self.next_button, stretch=1)
        hlayout2.addWidget(self.idx_string, stretch=2)
        self.interaction_layout1 = QtWidgets.QVBoxLayout()
        self.interaction_layout1.addLayout(hlayout1)
        self.interaction_layout1.addLayout(hlayout2)
        # Group 2
        self.interaction_layout2 = QtWidgets.QHBoxLayout()
        self.interaction_layout2.addWidget(self.reset_button)
        self.interaction_layout2.addWidget(self.complete_button)
        # Group 3
        self.interaction_layout3 = QtWidgets.QHBoxLayout()
        self.interaction_layout3.addWidget(subj_combobox, stretch=1)
        self.interaction_layout3.addWidget(sess_combobox, stretch=2)
        self.interaction_layout3.addWidget(self.data_button, stretch=1)

    def init_figures(self):
        """
        Create all figures that will be added to the GUI
        """
        # Figures to show ephys data
        # 2D scatter/ image plot
        self.fig_img = pg.PlotItem()
        self.fig_img.setMouseEnabled(x=False, y=False)
        self.fig_img.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                               self.probe_extra, padding=self.pad)
        self.fig_img.addLine(y=self.probe_tip, pen=self.kpen_dot, z=50)
        self.fig_img.addLine(y=self.probe_top, pen=self.kpen_dot, z=50)
        self.set_axis(self.fig_img, 'bottom')
        self.fig_data_ax = self.set_axis(self.fig_img, 'left',
                                         label='Distance from probe tip (uV)')

        self.fig_img_cb = pg.PlotItem()
        self.fig_img_cb.setMaximumHeight(70)
        self.fig_img_cb.setMouseEnabled(x=False, y=False)
        self.set_axis(self.fig_img_cb, 'bottom', show=False)
        self.set_axis(self.fig_img_cb, 'left', pen='w')
        self.set_axis(self.fig_img_cb, 'top', pen='w')

        # 1D line plot
        self.fig_line = pg.PlotItem()
        self.fig_line.setMouseEnabled(x=False, y=False)
        self.fig_line.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                self.probe_extra, padding=self.pad)
        self.fig_line.addLine(y=self.probe_tip, pen=self.kpen_dot, z=50)
        self.fig_line.addLine(y=self.probe_top, pen=self.kpen_dot, z=50)
        self.set_axis(self.fig_line, 'bottom')
        self.set_axis(self.fig_line, 'left', show=False)

        # 2D probe plot
        self.fig_probe = pg.PlotItem()
        self.fig_probe.setMouseEnabled(x=False, y=False)
        self.fig_probe.setMaximumWidth(50)
        self.fig_probe.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                 self.probe_extra, padding=self.pad)
        self.fig_probe.addLine(y=self.probe_tip, pen=self.kpen_dot, z=50)
        self.fig_probe.addLine(y=self.probe_top, pen=self.kpen_dot, z=50)
        self.set_axis(self.fig_probe, 'bottom', pen='w')
        self.set_axis(self.fig_probe, 'left', show=False)

        self.fig_probe_cb = pg.PlotItem()
        self.fig_probe_cb.setMouseEnabled(x=False, y=False)
        self.fig_probe_cb.setMaximumHeight(70)
        self.set_axis(self.fig_probe_cb, 'bottom', show=False)
        self.set_axis(self.fig_probe_cb, 'left', show=False)
        self.set_axis(self.fig_probe_cb, 'top', pen='w')

        # Add img plot, line plot, probe plot, img colourbar and probe colourbar to a graphics
        # layout widget so plots can be arranged and moved easily
        self.fig_data_area = pg.GraphicsLayoutWidget()
        self.fig_data_area.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_data_area.scene().sigMouseHover.connect(self.on_mouse_hover)
        self.fig_data_layout = pg.GraphicsLayout()

        self.fig_data_layout.addItem(self.fig_img_cb, 0, 0)
        self.fig_data_layout.addItem(self.fig_probe_cb, 0, 1, 1, 2)
        self.fig_data_layout.addItem(self.fig_img, 1, 0)
        self.fig_data_layout.addItem(self.fig_line, 1, 1)
        self.fig_data_layout.addItem(self.fig_probe, 1, 2)
        self.fig_data_layout.layout.setColumnStretchFactor(0, 6)
        self.fig_data_layout.layout.setColumnStretchFactor(1, 2)
        self.fig_data_layout.layout.setColumnStretchFactor(2, 1)
        self.fig_data_layout.layout.setRowStretchFactor(0, 1)
        self.fig_data_layout.layout.setRowStretchFactor(1, 10)

        self.fig_data_area.addItem(self.fig_data_layout)

        # Figures to show histology data
        # Histology figure that will be updated with user input
        self.fig_hist = pg.PlotItem()
        self.fig_hist.setContentsMargins(0, 0, 0, 0)
        self.fig_hist.setMouseEnabled(x=False)
        self.fig_hist.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                self.probe_extra, padding=self.pad)
        self.set_axis(self.fig_hist, 'bottom', pen='w', ticks=False)
        self.ax_hist = self.set_axis(self.fig_hist, 'left', pen=None)
        self.ax_hist.setWidth(0)
        self.ax_hist.setStyle(tickTextOffset=-70)

        self.fig_stretch = pg.PlotItem()
        self.fig_stretch.setMaximumWidth(50)
        self.fig_stretch.setMouseEnabled(x=False)
        self.set_axis(self.fig_stretch, 'bottom', pen='w', ticks=False)
        self.set_axis(self.fig_stretch, 'left', show=False)
        (self.fig_stretch).setYLink(self.fig_hist)

        # Figure that will show scale factor of histology boundaries
        self.fig_stretch_cb = pg.PlotItem()
        self.fig_stretch_cb.setMouseEnabled(x=False, y=False)
        self.fig_stretch_cb.setMaximumHeight(70)
        self.set_axis(self.fig_stretch_cb, 'bottom', show=False)
        self.set_axis(self.fig_stretch_cb, 'left', show=False)
        self.set_axis(self.fig_stretch_cb, 'top', pen='w')
        self.set_axis(self.fig_stretch_cb, 'right', show=False)

        # Histology figure that will remain at initial state for reference
        self.fig_hist_ref = pg.PlotItem()
        self.fig_hist_ref.setMouseEnabled(x=False, y=False)
        self.fig_hist_ref.setYRange(min=self.probe_tip - self.probe_extra, max=self.probe_top +
                                    self.probe_extra, padding=self.pad)
        self.set_axis(self.fig_hist_ref, 'bottom', pen='w', ticks=False)
        self.set_axis(self.fig_hist_ref, 'left', show=False)
        self.ax_hist_ref = self.set_axis(self.fig_hist_ref, 'right', pen=None)
        self.ax_hist_ref.setWidth(0)
        self.ax_hist_ref.setStyle(tickTextOffset=-70)

        self.fig_hist_area = pg.GraphicsLayoutWidget()
        self.fig_hist_area.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_hist_area.scene().sigMouseHover.connect(self.on_mouse_hover)
        self.fig_hist_layout = pg.GraphicsLayout()
        self.fig_hist_layout.addItem(self.fig_stretch_cb, 0, 0, 1, 3)
        self.fig_hist_layout.addItem(self.fig_hist, 1, 0)
        self.fig_hist_layout.addItem(self.fig_stretch, 1, 1)
        self.fig_hist_layout.addItem(self.fig_hist_ref, 1, 2,)
        self.fig_hist_layout.layout.setColumnStretchFactor(0, 4)
        self.fig_hist_layout.layout.setColumnStretchFactor(1, 1)
        self.fig_hist_layout.layout.setColumnStretchFactor(2, 4)
        self.fig_hist_layout.layout.setRowStretchFactor(0, 1)
        self.fig_hist_layout.layout.setRowStretchFactor(1, 10)
        self.fig_hist_area.addItem(self.fig_hist_layout)

        # Figure to show probe location through coronal slice of brain
        self.fig_slice = matplot.MatplotlibWidget()
        fig = self.fig_slice.getFigure()
        fig.canvas.toolbar.hide()
        self.fig_slice_ax = fig.gca()
        self.fig_slice_ax.axis('off')

        # Figure to show fit and offset applied by user
        self.fig_fit = pg.PlotWidget(background='w')
        self.fig_fit.setMouseEnabled(x=False, y=False)
        self.fig_fit.setXRange(min=self.view_total[0], max=self.view_total[1])
        self.fig_fit.setYRange(min=self.view_total[0], max=self.view_total[1])
        self.set_axis(self.fig_fit, 'bottom')
        self.set_axis(self.fig_fit, 'left')
        plot = pg.PlotCurveItem()
        plot.setData(x=self.depth, y=self.depth, pen=self.kpen_dot)
        self.fit_plot = pg.PlotCurveItem(pen=self.bpen_solid)
        self.fig_fit.addItem(plot)
        self.fig_fit.addItem(self.fit_plot)
