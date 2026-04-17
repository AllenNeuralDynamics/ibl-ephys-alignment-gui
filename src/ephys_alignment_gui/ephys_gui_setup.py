from random import randrange

import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters
from PyQt5 import QtCore, QtGui, QtWidgets

from ephys_alignment_gui.plot_elements import replace_axis

pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")


class Setup:
    def init_layout(self, main_window, offline=False) -> None:
        self.resize(1600, 800)
        self.setWindowTitle("IBL Ephys Alignment GUI")
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.offline = offline
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)

        self.init_interaction_features()
        self.init_figures()

        main_layout = QtWidgets.QGridLayout()
        main_layout.addWidget(self.fig_data_area, 0, 0, 10, 1)
        main_layout.addWidget(self.fig_hist_area, 0, 1, 10, 1)
        main_layout.addLayout(self.interaction_layout1, 0, 2, 2, 1)
        main_layout.addWidget(self.fig_slice_area, 2, 2, 2, 1)
        main_layout.addLayout(self.interaction_layout2, 4, 2, 2, 1)
        main_layout.addWidget(self.fig_fit, 6, 2, 2, 1)
        main_layout.addLayout(self.interaction_layout3, 8, 2, 2, 1)
        main_layout.setColumnStretch(0, 5)
        main_layout.setColumnStretch(1, 2)
        main_layout.setColumnStretch(2, 3)

        main_widget.setLayout(main_layout)

    def init_menubar(self) -> None:
        """
        Create menu bar and add all possible menu options. These are:
            - Image Plots: possible 2D image/scatter plots
            - Line Plots: possible 1D line plots
            - Probe Plots: possible 2D plots arranged according to probe geometry
            - Slice Plots: possible coronal slice images
            - Filter Units: filter displayed plots by unit type (All, Good, MUA)
            - Fit Options: possible keyboard interactions for applying alignment
            - Display Options: possible keyboard interactions to what is displayed on GUI
            - Session Information: extra info, session notes and Allen brain regions description
        """
        # Create menubar widget and add it to the main GUI window
        menu_bar = QtWidgets.QMenuBar(self)
        menu_bar.setNativeMenuBar(False)
        self.setMenuBar(menu_bar)

        # IMAGE PLOTS MENU BAR
        # Define all 2D scatter/ image plot options
        scatter_drift = QtWidgets.QAction(
            "Amplitude", self, checkable=True, checked=False
        )
        scatter_drift.triggered.connect(lambda: self.plot_scatter(self.session.scat_drift_data))
        scatter_fr = QtWidgets.QAction(
            "Cluster Amp vs Depth vs FR", self, checkable=True, checked=False
        )
        scatter_fr.triggered.connect(lambda: self.plot_scatter(self.session.scat_fr_data))
        scatter_p2t = QtWidgets.QAction(
            "Cluster Amp vs Depth vs Duration",
            self,
            checkable=True,
            checked=False,
        )
        scatter_p2t.triggered.connect(lambda: self.plot_scatter(self.session.scat_p2t_data))
        scatter_amp = QtWidgets.QAction(
            "Cluster FR vs Depth vs Amp", self, checkable=True, checked=False
        )
        scatter_amp.triggered.connect(lambda: self.plot_scatter(self.session.scat_amp_data))
        img_fr = QtWidgets.QAction("Firing Rate", self, checkable=True, checked=True)
        img_fr.triggered.connect(lambda: self.plot_image(self.session.img_fr_data))
        img_spike_corr = QtWidgets.QAction(
            "Spike Correlation", self, checkable=True, checked=False
        )
        img_spike_corr.triggered.connect(
            lambda: self.plot_image(self.session.img_spike_corr_data)
        )
        img_rmsAP = QtWidgets.QAction("RMS AP", self, checkable=True, checked=False)
        img_rmsAP.triggered.connect(lambda: self.plot_image(self.session.img_rms_APdata))
        img_rmsLFP = QtWidgets.QAction("RMS LFP", self, checkable=True, checked=False)
        img_rmsLFP.triggered.connect(lambda: self.plot_image(self.session.img_rms_LFPdata))
        img_LFP = QtWidgets.QAction("LFP Spectrum", self, checkable=True, checked=False)
        img_LFP.triggered.connect(lambda: self.plot_image(self.session.img_lfp_data))

        img_lfp_corr_list = []
        for lfp_corr_key in self.session.img_lfp_corr_data:
            img_lfp_corr = QtWidgets.QAction(
                f"LFP Correlation ({lfp_corr_key})",
                self,
                checkable=True,
                checked=False,
            )
            img_lfp_corr.triggered.connect(
                lambda checked, key=lfp_corr_key: self.plot_image(
                    self.session.img_lfp_corr_data[key]
                )
            )
            img_lfp_corr_list.append(img_lfp_corr)

        img_rmsAP_main = QtWidgets.QAction(
            "RMS AP Main Rec", self, checkable=True, checked=False
        )
        img_rmsAP_main.triggered.connect(
            lambda: self.plot_image(self.session.img_rms_APdata_main)
        )
        img_rmsLFP_main = QtWidgets.QAction(
            "RMS LFP Main Rec", self, checkable=True, checked=False
        )
        img_rmsLFP_main.triggered.connect(
            lambda: self.plot_image(self.session.img_rms_LFPdata_main)
        )
        img_LFP_main = QtWidgets.QAction(
            "LFP Spectrum Main Rec", self, checkable=True, checked=False
        )
        img_LFP_main.triggered.connect(lambda: self.plot_image(self.session.img_lfp_data_main))

        # Initialise with firing rate 2D plot
        self.img_init = img_fr

        # Add menu bar for 2D scatter/ image plot options
        img_options = menu_bar.addMenu("Image Plots")
        # Add action group so we can toggle through 2D scatter/ image plot options
        self.img_options_group = QtWidgets.QActionGroup(img_options)
        # Only allow one to plot to be selected at any one time
        self.img_options_group.setExclusive(True)
        img_options.addAction(img_fr)
        self.img_options_group.addAction(img_fr)
        img_options.addAction(scatter_drift)
        self.img_options_group.addAction(scatter_drift)
        img_options.addAction(img_spike_corr)
        self.img_options_group.addAction(img_spike_corr)
        img_options.addAction(img_rmsAP)
        self.img_options_group.addAction(img_rmsAP)
        img_options.addAction(img_rmsAP_main)
        self.img_options_group.addAction(img_rmsAP_main)
        img_options.addAction(img_rmsLFP)
        self.img_options_group.addAction(img_rmsLFP)

        for img_lfp_corr in img_lfp_corr_list:
            img_options.addAction(img_lfp_corr)
            self.img_options_group.addAction(img_lfp_corr)

        img_options.addAction(img_rmsLFP_main)
        self.img_options_group.addAction(img_rmsLFP_main)
        img_options.addAction(img_LFP)
        self.img_options_group.addAction(img_LFP)
        img_options.addAction(img_LFP_main)
        self.img_options_group.addAction(img_LFP_main)
        img_options.addAction(scatter_fr)
        self.img_options_group.addAction(scatter_fr)
        img_options.addAction(scatter_p2t)
        self.img_options_group.addAction(scatter_p2t)
        img_options.addAction(scatter_amp)
        self.img_options_group.addAction(scatter_amp)

        raw_type = list(self.session.img_raw_data.keys())
        for raw in raw_type:
            img = QtWidgets.QAction(raw, self, checkable=True, checked=False)
            img.triggered.connect(
                lambda checked, item=raw: self.plot_image(self.session.img_raw_data[item])
            )
            img_options.addAction(img)
            self.img_options_group.addAction(img)

        stim_type = list(self.session.img_stim_data.keys())
        for stim in stim_type:
            img = QtWidgets.QAction(stim, self, checkable=True, checked=False)
            img.triggered.connect(
                lambda checked, item=stim: self.plot_image(self.session.img_stim_data[item])
            )
            img_options.addAction(img)
            self.img_options_group.addAction(img)

        self.img_options_group.triggered.connect(self._on_img_action_triggered)
        self.current_img_action = self.img_init
        self.current_img_action.setChecked(
            True
        )  # makes sure menu reflects initial plot

        # LINE PLOTS MENU BAR
        # Define all 1D line plot options
        line_fr = QtWidgets.QAction("Firing Rate", self, checkable=True, checked=True)
        line_fr.triggered.connect(lambda: self.plot_line(self.session.line_fr_data))
        line_amp = QtWidgets.QAction("Amplitude", self, checkable=True, checked=False)
        line_amp.triggered.connect(lambda: self.plot_line(self.session.line_amp_data))
        # Initialise with firing rate 1D plot
        self.line_init = line_fr
        # Add menu bar for 1D line plot options
        line_options = menu_bar.addMenu("Line Plots")
        # Add action group so we can toggle through 2D scatter/ image plot options
        self.line_options_group = QtWidgets.QActionGroup(line_options)
        # Only allow one to plot to be selected at any one time
        self.line_options_group.setExclusive(True)
        line_options.addAction(line_fr)
        self.line_options_group.addAction(line_fr)
        line_options.addAction(line_amp)
        self.line_options_group.addAction(line_amp)

        self.line_options_group.triggered.connect(self._on_line_action_triggered)
        self.line_img_action = self.line_init
        self.line_img_action.setChecked(True)  # makes sure menu reflects initial plot

        # PROBE PLOTS MENU BAR
        # Define all 2D probe plot options
        # In two stages 1) RMS plots manually, 2) frequency plots in for loop
        probe_rmsAP = QtWidgets.QAction("RMS AP", self, checkable=True, checked=True)
        probe_rmsAP.triggered.connect(lambda: self.plot_probe(self.session.probe_rms_APdata))
        probe_rmsLFP = QtWidgets.QAction("RMS LFP", self, checkable=True, checked=False)
        probe_rmsLFP.triggered.connect(lambda: self.plot_probe(self.session.probe_rms_LFPdata))

        # Initialise with rms of AP probe plot
        self.probe_init = probe_rmsAP

        # Add menu bar for 2D probe plot options
        probe_options = menu_bar.addMenu("Probe Plots")
        # Add action group so we can toggle through probe plot options
        self.probe_options_group = QtWidgets.QActionGroup(probe_options)
        self.probe_options_group.setExclusive(True)
        probe_options.addAction(probe_rmsAP)
        self.probe_options_group.addAction(probe_rmsAP)
        probe_options.addAction(probe_rmsLFP)
        self.probe_options_group.addAction(probe_rmsLFP)

        self.probe_options_group.triggered.connect(self._on_probe_action_triggered)
        self.probe_img_action = self.probe_init
        self.probe_img_action.setChecked(True)  # makes sure menu reflects initial plot

        # Add the different frequency band options in a loop. These bands must be the same as
        # defined in plot_data
        freq_bands = np.vstack(([0, 4], [4, 10], [10, 30], [30, 80], [80, 200]))
        for iF, freq in enumerate(freq_bands):
            band = f"{freq[0]} - {freq[1]} Hz"
            probe = QtWidgets.QAction(band, self, checkable=True, checked=False)
            probe.triggered.connect(
                lambda checked, item=band: self.plot_probe(self.session.probe_lfp_data[item])
            )
            probe_options.addAction(probe)
            self.probe_options_group.addAction(probe)

        sub_types = list(self.session.probe_rfmap.keys())
        for sub in sub_types:
            probe = QtWidgets.QAction(
                f"RF Map - {sub}", self, checkable=True, checked=False
            )
            probe.triggered.connect(
                lambda checked, item=sub: self.plot_probe(
                    self.session.probe_rfmap[item], bounds=self.session.rfmap_boundaries
                )
            )
            probe_options.addAction(probe)
            self.probe_options_group.addAction(probe)

        # SLICE PLOTS MENU BAR
        self.init_slice_menu()

        # FILTER UNITS MENU BAR
        # Define unit filtering options
        all_units = QtWidgets.QAction("All", self, checkable=True, checked=True)
        all_units.triggered.connect(lambda: self.filter_unit_pressed("all"))
        good_units = QtWidgets.QAction("KS good", self, checkable=True, checked=False)
        good_units.triggered.connect(lambda: self.filter_unit_pressed("KS good"))
        mua_units = QtWidgets.QAction("KS mua", self, checkable=True, checked=False)
        mua_units.triggered.connect(lambda: self.filter_unit_pressed("KS mua"))
        ibl_units = QtWidgets.QAction("IBL good", self, checkable=True, checked=False)
        ibl_units.triggered.connect(lambda: self.filter_unit_pressed("IBL good"))
        aind_qc_units = QtWidgets.QAction(
            "aind_qc", self, checkable=True, checked=False
        )
        aind_qc_units.triggered.connect(lambda: self.filter_unit_pressed("aind_qc"))
        sua_refine_units = QtWidgets.QAction(
            "unitrefine_sua", self, checkable=True, checked=False
        )
        sua_refine_units.triggered.connect(
            lambda: self.filter_unit_pressed("unitrefine_sua")
        )
        neural_refine_units = QtWidgets.QAction(
            "unitrefine_neural", self, checkable=True, checked=False
        )
        neural_refine_units.triggered.connect(
            lambda: self.filter_unit_pressed("unitrefine_neural")
        )
        # Initialise with all units being shown
        self.unit_init = all_units

        # Add menu bar for slice plot options
        unit_filter_options = menu_bar.addMenu("Filter Units")
        # Add action group so we can toggle through unit options
        self.unit_filter_options_group = QtWidgets.QActionGroup(unit_filter_options)
        self.unit_filter_options_group.setExclusive(True)
        unit_filter_options.addAction(all_units)
        self.unit_filter_options_group.addAction(all_units)
        unit_filter_options.addAction(good_units)
        self.unit_filter_options_group.addAction(good_units)
        unit_filter_options.addAction(mua_units)
        self.unit_filter_options_group.addAction(mua_units)
        unit_filter_options.addAction(ibl_units)
        self.unit_filter_options_group.addAction(ibl_units)
        unit_filter_options.addAction(aind_qc_units)
        self.unit_filter_options_group.addAction(aind_qc_units)
        unit_filter_options.addAction(sua_refine_units)
        self.unit_filter_options_group.addAction(sua_refine_units)
        unit_filter_options.addAction(neural_refine_units)
        self.unit_filter_options_group.addAction(neural_refine_units)

        # FIT OPTIONS MENU BAR
        # Define all possible keyboard shortcut interactions for GUI

        # Shortcut to apply interpolation
        fit_option = QtWidgets.QAction("Fit", self)
        fit_option.setShortcut("Return")
        fit_option.triggered.connect(self.fit_button_pressed)

        # Shortcuts to apply offset
        offset_option = QtWidgets.QAction("Offset", self)
        offset_option.setShortcut("O")
        offset_option.triggered.connect(self.offset_button_pressed)
        moveup_option = QtWidgets.QAction("Offset + 50um", self)
        moveup_option.setShortcut("Shift+Up")
        moveup_option.triggered.connect(self.moveup_button_pressed)
        movedown_option = QtWidgets.QAction("Offset - 50um", self)
        movedown_option.setShortcut("Shift+Down")
        movedown_option.triggered.connect(self.movedown_button_pressed)

        # Shortcut to delete a reference line
        delete_line_option = QtWidgets.QAction("Delete Line", self)
        delete_line_option.setShortcut("Shift+D")
        delete_line_option.triggered.connect(self.delete_line_button_pressed)

        # Shortcut to move between previous/next moves
        next_option = QtWidgets.QAction("Next", self)
        next_option.setShortcut("Right")
        next_option.triggered.connect(self.next_button_pressed)
        prev_option = QtWidgets.QAction("Previous", self)
        prev_option.setShortcut("Left")
        prev_option.triggered.connect(self.prev_button_pressed)

        # Shortcut to reset GUI to initial state
        reset_option = QtWidgets.QAction("Reset", self)
        reset_option.setShortcut("Ctrl+R")
        reset_option.triggered.connect(self.reset_button_pressed)

        # Shortcut to save final state to JSON file
        complete_option = QtWidgets.QAction("Save", self)
        complete_option.setShortcut("Ctrl+S")
        if not self.offline:
            complete_option.triggered.connect(self.display_qc_options)
        else:
            complete_option.triggered.connect(self.complete_button_pressed_offline)

        # Add menu bar with all possible keyboard interactions
        fit_options = menu_bar.addMenu("Fit Options")
        fit_options.addAction(fit_option)
        fit_options.addAction(offset_option)
        fit_options.addAction(moveup_option)
        fit_options.addAction(movedown_option)
        fit_options.addAction(delete_line_option)
        fit_options.addAction(next_option)
        fit_options.addAction(prev_option)
        fit_options.addAction(reset_option)
        fit_options.addAction(complete_option)

        # DISPLAY OPTIONS MENU BAR
        # Define all possible keyboard shortcut for visualisation features
        # Shortcuts to toggle between plots options
        toggle1_option = QtWidgets.QAction("Toggle Image Plots", self)
        toggle1_option.setShortcut("Alt+1")
        toggle1_option.triggered.connect(
            lambda: self.toggle_plots(self.img_options_group)
        )
        toggle2_option = QtWidgets.QAction("Toggle Line Plots", self)
        toggle2_option.setShortcut("Alt+2")
        toggle2_option.triggered.connect(
            lambda: self.toggle_plots(self.line_options_group)
        )
        toggle3_option = QtWidgets.QAction("Toggle Probe Plots", self)
        toggle3_option.setShortcut("Alt+3")
        toggle3_option.triggered.connect(
            lambda: self.toggle_plots(self.probe_options_group)
        )
        toggle4_option = QtWidgets.QAction("Toggle Slice Plots", self)
        toggle4_option.setShortcut("Alt+4")
        toggle4_option.triggered.connect(
            lambda: self.toggle_plots(self.slice_options_group)
        )

        toggle5_option = QtWidgets.QAction("Toggle Previous Image Plots", self)
        toggle5_option.setShortcut("Alt+Ctrl+1")
        toggle5_option.triggered.connect(
            lambda: self.toggle_plots(self.img_options_group, reverse=True)
        )
        toggle6_option = QtWidgets.QAction("Toggle Previous Line Plots", self)
        toggle6_option.setShortcut("Alt+Ctrl+2")
        toggle6_option.triggered.connect(
            lambda: self.toggle_plots(self.line_options_group, reverse=True)
        )
        toggle7_option = QtWidgets.QAction("Toggle Previous Probe Plots", self)
        toggle7_option.setShortcut("Alt+Ctrl+3")
        toggle7_option.triggered.connect(
            lambda: self.toggle_plots(self.probe_options_group, reverse=True)
        )
        toggle8_option = QtWidgets.QAction("Toggle Previous Slice Plots", self)
        toggle8_option.setShortcut("Alt+Ctrl+4")
        toggle8_option.triggered.connect(
            lambda: self.toggle_plots(self.slice_options_group, reverse=True)
        )

        # Shortcuts to switch order of 3 panels in ephys plot
        view1_option = QtWidgets.QAction("View 1", self)
        view1_option.setShortcut("Shift+1")
        view1_option.triggered.connect(lambda: self.set_view(view=1))
        view2_option = QtWidgets.QAction("View 2", self)
        view2_option.setShortcut("Shift+2")
        view2_option.triggered.connect(lambda: self.set_view(view=2))
        view3_option = QtWidgets.QAction("View 3", self)
        view3_option.setShortcut("Shift+3")
        view3_option.triggered.connect(lambda: self.set_view(view=3))

        # Shortcut to reset axis on figures
        axis_option = QtWidgets.QAction("Reset Axis", self)
        axis_option.setShortcut("Shift+A")
        axis_option.triggered.connect(self.reset_axis_button_pressed)

        # Shortcut to hide/show region labels
        toggle_labels_option = QtWidgets.QAction("Hide/Show Labels", self)
        toggle_labels_option.setShortcut("Shift+L")
        toggle_labels_option.triggered.connect(self.toggle_labels_button_pressed)

        # Shortcut to hide/show reference lines
        toggle_lines_option = QtWidgets.QAction("Hide/Show Lines", self)
        toggle_lines_option.setShortcut("Shift+H")
        toggle_lines_option.triggered.connect(self.toggle_line_button_pressed)

        # Shortcut to hide/show reference lines and channels on slice image
        toggle_channels_option = QtWidgets.QAction("Hide/Show Channels", self)
        toggle_channels_option.setShortcut("Shift+C")
        toggle_channels_option.triggered.connect(self.toggle_channel_button_pressed)

        # Shortcut to change default histology reference image
        toggle_histology_option = QtWidgets.QAction("Hide/Show Nearby Boundaries", self)
        toggle_histology_option.setShortcut("Shift+N")
        toggle_histology_option.triggered.connect(self.toggle_histology_button_pressed)

        # Option to change histology regions from Allen to Franklin Paxinos
        toggle_histology_map_option = QtWidgets.QAction("Change Histology Map", self)
        toggle_histology_map_option.setShortcut("Shift+M")
        toggle_histology_map_option.triggered.connect(
            self.toggle_histology_map_button_pressed
        )

        # Shortcuts for cluster popup window
        popup_minimise = QtWidgets.QAction("Minimise/Show Cluster Popup", self)
        popup_minimise.setShortcut("Alt+M")
        popup_minimise.triggered.connect(self.minimise_popups)
        popup_close = QtWidgets.QAction("Close Cluster Popup", self)
        popup_close.setShortcut("Alt+X")
        popup_close.triggered.connect(self.close_popups)

        # Option to save all plots
        save_plots = QtWidgets.QAction("Save Plots", self)
        save_plots.setShortcut("Ctrl+Shift+S")
        save_plots.triggered.connect(self.save_plots)

        # Add menu bar with all possible display options
        display_options = menu_bar.addMenu("Display Options")
        display_options.addAction(toggle1_option)
        display_options.addAction(toggle2_option)
        display_options.addAction(toggle3_option)
        display_options.addAction(toggle4_option)
        display_options.addAction(toggle5_option)
        display_options.addAction(toggle6_option)
        display_options.addAction(toggle7_option)
        display_options.addAction(toggle8_option)
        display_options.addAction(view1_option)
        display_options.addAction(view2_option)
        display_options.addAction(view3_option)
        display_options.addAction(axis_option)
        display_options.addAction(toggle_labels_option)
        display_options.addAction(toggle_lines_option)
        display_options.addAction(toggle_channels_option)
        display_options.addAction(toggle_histology_option)
        display_options.addAction(toggle_histology_map_option)
        display_options.addAction(popup_minimise)
        display_options.addAction(popup_close)
        display_options.addAction(save_plots)

        # SESSION INFORMATION MENU BAR
        # Define all session information options
        # Display any notes associated with recording session
        session_notes = QtWidgets.QAction("Session Notes", self)
        session_notes.triggered.connect(self.display_session_notes)
        # Shortcut to show label information
        region_info = QtWidgets.QAction("Region Info", self)
        region_info.setShortcut("Shift+I")
        region_info.triggered.connect(self.describe_labels_pressed)

        # Add menu bar with all possible session info options
        info_options = menu_bar.addMenu("Session Information")
        info_options.addAction(session_notes)
        info_options.addAction(region_info)

        # Display other sessions that are closeby if online mode
        if not self.offline:
            nearby_info = QtWidgets.QAction("Nearby Sessions", self)
            nearby_info.triggered.connect(self.display_nearby_sessions)
            info_options.addAction(nearby_info)

            scaling_info = QtWidgets.QAction("Subject Scaling", self)
            scaling_info.triggered.connect(self.display_subject_scaling)
            info_options.addAction(scaling_info)

            feature_info = QtWidgets.QAction("Region Feature", self)
            feature_info.triggered.connect(self.display_region_features)
            info_options.addAction(feature_info)

    def init_slice_menu(self) -> None:
        menu_bar = self.menuBar()
        # Define all coronal slice plot options. Each action stores
        # (session_attr_name, slice_data_key) via setData() so callers can
        # later look up the data source without re-deriving it from the label.
        slice_ccf = QtWidgets.QAction("CCF", self, checkable=True, checked=False)
        slice_ccf.setData(("slice_data", "ccf"))
        slice_ccf.triggered.connect(lambda: self.plot_slice(self.session.slice_data, "ccf"))
        slice_label = QtWidgets.QAction(
            "Annotation", self, checkable=True, checked=False
        )
        slice_label.setData(("slice_data", "label"))
        slice_label.triggered.connect(lambda: self.plot_slice(self.session.slice_data, "label"))
        if self.session.fp_slice_data is not None:
            fp_slice_label = QtWidgets.QAction(
                "Annotation FP", self, checkable=True, checked=False
            )
            fp_slice_label.setData(("fp_slice_data", "label"))
            fp_slice_label.triggered.connect(
                lambda: self.plot_slice(self.session.fp_slice_data, "label")
            )
        if not self.offline:
            slice_hist_cb = QtWidgets.QAction(
                "Histology cerebellar example",
                self,
                checkable=True,
                checked=False,
            )
            slice_hist_cb.setData(("slice_data", "hist_cb"))
            slice_hist_cb.triggered.connect(
                lambda: self.plot_slice(self.session.slice_data, "hist_cb")
            )

        # Add menu bar for slice plot
        slice_options = menu_bar.addMenu("Slice Plots")
        # Add action group so we can toggle through slice plot options
        self.slice_options_group = QtWidgets.QActionGroup(slice_options)
        self.slice_options_group.setExclusive(True)
        slice_options.addAction(slice_ccf)
        self.slice_options_group.addAction(slice_ccf)
        slice_options.addAction(slice_label)
        self.slice_options_group.addAction(slice_label)
        if self.session.fp_slice_data is not None:
            slice_options.addAction(fp_slice_label)
            self.slice_options_group.addAction(fp_slice_label)

        if not self.offline:
            slice_options.addAction(slice_hist_cb)
            self.slice_options_group.addAction(slice_hist_cb)

        # Initialise with the CCF as the default slice plot.
        self.slice_init = slice_ccf

        # These are accessed sepperatly so have conditional activation
        for key in self.session.slice_data.keys():
            if key in ["ccf", "label", "scale", "offset"]:
                continue
            else:
                this_slice_action = QtWidgets.QAction(
                    key, self, checkable=True, checked=False
                )
                this_slice_action.setData(("slice_data", key))
                this_slice_action.triggered.connect(
                    lambda checked, item=key: self.plot_slice(self.session.slice_data, item)
                )
                slice_options.addAction(this_slice_action)
                self.slice_options_group.addAction(this_slice_action)

    def init_interaction_features(self) -> None:
        """
        Create all interaction widgets that will be added to the GUI
        """
        # Button to apply interpolation
        self.fit_button = QtWidgets.QPushButton("Fit")
        self.fit_button.clicked.connect(self.fit_button_pressed)
        # Button to apply offset
        self.offset_button = QtWidgets.QPushButton("Offset")
        self.offset_button.clicked.connect(self.offset_button_pressed)
        # Button to go to next move
        self.next_button = QtWidgets.QPushButton("Next")
        self.next_button.clicked.connect(self.next_button_pressed)
        # Button to go to previous move
        self.prev_button = QtWidgets.QPushButton("Previous")
        self.prev_button.clicked.connect(self.prev_button_pressed)
        # String to display current move index
        self.idx_string = QtWidgets.QLabel()
        # String to display total number of moves
        self.tot_idx_string = QtWidgets.QLabel()
        # Button to reset GUI to initial state
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_button_pressed)
        # Button to upload final state to Alyx/ to local file
        self.complete_button = QtWidgets.QPushButton("Save")
        if not self.offline:
            self.complete_button.clicked.connect(self.display_qc_options)
        else:
            self.complete_button.clicked.connect(self.complete_button_pressed_offline)

        if not self.offline:
            # If offline mode is False, read in Subject and Session options from Alyx
            # Drop down list to choose subject
            self.subj_list = QtGui.QStandardItemModel()
            self.subj_combobox = QtWidgets.QComboBox()
            # Add line edit and completer to be able to search for subject
            self.subj_combobox.setLineEdit(QtWidgets.QLineEdit())
            subj_completer = QtWidgets.QCompleter()
            subj_completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
            self.subj_combobox.setCompleter(subj_completer)
            self.subj_combobox.setModel(self.subj_list)
            self.subj_combobox.completer().setModel(self.subj_list)
            self.subj_combobox.activated.connect(self.on_subject_selected)

            # Drop down list to choose session
            self.sess_list = QtGui.QStandardItemModel()
            self.sess_combobox = QtWidgets.QComboBox()
            self.sess_combobox.setModel(self.sess_list)
            self.sess_combobox.activated.connect(self.on_session_selected)
        else:
            # Offline mode: user points the GUI at a preprocessed mouse-root
            # directory (containing ``datapackage.json``) and selects the
            # session + probe from dropdowns populated from the manifest.
            self.mouse_root_line = QtWidgets.QLineEdit()
            self.mouse_root_button = QtWidgets.QToolButton()
            self.mouse_root_button.setText("Mouse Root")
            self.mouse_root_button.clicked.connect(self.on_mouse_root_selected)
            self.mouse_root_line.editingFinished.connect(self.on_mouse_root_edited)

            self.session_list = QtGui.QStandardItemModel()
            self.session_combobox = QtWidgets.QComboBox()
            self.session_combobox.setModel(self.session_list)
            self.session_combobox.activated.connect(self.on_session_combobox_activated)

            self.probe_list = QtGui.QStandardItemModel()
            self.probe_combobox = QtWidgets.QComboBox()
            self.probe_combobox.setModel(self.probe_list)
            self.probe_combobox.activated.connect(self.on_probe_combobox_activated)

            self.reload_folder_line = QtWidgets.QLineEdit()
            self.reload_folder_button = QtWidgets.QToolButton()
            self.reload_folder_button.setText("Load Alignments")
            self.reload_folder_button.clicked.connect(self.load_existing_alignments)

        # Drop down list to select shank
        self.shank_list = QtGui.QStandardItemModel()
        self.shank_combobox = QtWidgets.QComboBox()
        self.shank_combobox.setModel(self.shank_list)
        self.shank_combobox.activated.connect(self.on_shank_selected)

        # Drop down list to select previous alignment (NEW)
        self.align_list = QtGui.QStandardItemModel()
        self.align_combobox = QtWidgets.QComboBox()
        self.align_combobox.setModel(self.align_list)
        self.align_combobox.activated.connect(self.on_alignment_selected)

        self.output_folder_line = QtWidgets.QLineEdit()
        self.output_folder_button = QtWidgets.QToolButton()
        self.output_folder_button.setText("Output Directory")
        self.output_folder_button.clicked.connect(self.on_output_folder_selected)
        self.output_folder_line.editingFinished.connect(self.on_output_folder_edited)

        # After output_folder_button creation (around line 573):
        self.use_docdb_checkbox = QtWidgets.QCheckBox("DocDB")
        self.use_docdb_checkbox.setChecked(True)  # Default: try DocDB
        self.use_docdb_checkbox.stateChanged.connect(self.on_use_docdb_changed)

        self.load_data_button = QtWidgets.QToolButton()
        self.load_data_button.setText("Load Data")
        self.load_data_button.setEnabled(False)  # Disabled until input path is set
        self.load_data_button.clicked.connect(self.on_load_data_button_pressed)

        # Arrange interaction features into three different layout groups
        # Group 1 -- loading data
        if not self.offline:
            self.interaction_layout1 = QtWidgets.QHBoxLayout()
            self.interaction_layout1.addWidget(self.subj_combobox, stretch=1)
            self.interaction_layout1.addWidget(self.sess_combobox, stretch=2)
            self.interaction_layout1.addWidget(self.align_combobox, stretch=2)
            self.interaction_layout1.addWidget(self.data_button, stretch=1)
        else:
            interact_1_h_1 = QtWidgets.QHBoxLayout()
            interact_1_h_1.addWidget(self.mouse_root_button, stretch=0)
            interact_1_h_1.addWidget(self.mouse_root_line, stretch=2)

            interact_1_h_2 = QtWidgets.QHBoxLayout()
            interact_1_h_2.addWidget(self.session_combobox, stretch=2)
            interact_1_h_2.addWidget(self.probe_combobox, stretch=2)
            interact_1_h_2.addWidget(self.shank_combobox, stretch=1)
            interact_1_h_2.addWidget(self.load_data_button, stretch=0)

            interact_1_h_3 = QtWidgets.QHBoxLayout()
            interact_1_h_3.addWidget(self.align_combobox, stretch=1)
            interact_1_h_3.addWidget(self.use_docdb_checkbox, stretch=0)
            interact_1_h_3.addWidget(self.reload_folder_button, stretch=0)
            interact_1_h_3.addWidget(self.reload_folder_line, stretch=2)

            self.interaction_layout1 = QtWidgets.QVBoxLayout()
            self.interaction_layout1.addLayout(interact_1_h_1)
            self.interaction_layout1.addLayout(interact_1_h_2)
            self.interaction_layout1.addLayout(interact_1_h_3)

        # Group 2 -- fitting and navigation
        interact_2_h_1 = QtWidgets.QHBoxLayout()
        interact_2_h_1.addWidget(self.fit_button, stretch=1)
        interact_2_h_1.addWidget(self.offset_button, stretch=1)
        interact_2_h_1.addWidget(self.tot_idx_string, stretch=2)
        interact_2_h_2 = QtWidgets.QHBoxLayout()
        interact_2_h_2.addWidget(self.prev_button, stretch=1)
        interact_2_h_2.addWidget(self.next_button, stretch=1)
        interact_2_h_2.addWidget(self.idx_string, stretch=2)
        self.interaction_layout2 = QtWidgets.QVBoxLayout()
        self.interaction_layout2.addLayout(interact_2_h_1)
        self.interaction_layout2.addLayout(interact_2_h_2)

        # Group 3 -- saving data
        interact_3_h_1 = QtWidgets.QHBoxLayout()
        interact_3_h_1.addWidget(self.output_folder_button, stretch=0)
        interact_3_h_1.addWidget(self.output_folder_line, stretch=1)
        interact_3_h_2 = QtWidgets.QHBoxLayout()
        interact_3_h_2.addWidget(self.reset_button, stretch=0)
        interact_3_h_2.addWidget(self.complete_button, stretch=0)
        self.interaction_layout3 = QtWidgets.QVBoxLayout()
        self.interaction_layout3.addLayout(interact_3_h_1)
        self.interaction_layout3.addLayout(interact_3_h_2)

        # Pop up dialog for qc results to datajoint, only for online mode
        if not self.offline:
            align_qc_label = QtWidgets.QLabel("Confidence of alignment")
            self.align_qc = QtWidgets.QComboBox()
            self.align_qc.addItems(["High", "Medium", "Low"])
            ephys_qc_label = QtWidgets.QLabel("QC for ephys recording")
            self.ephys_qc = QtWidgets.QComboBox()
            self.ephys_qc.addItems(["Pass", "Warning", "Critical"])

            self.desc_buttons = QtWidgets.QButtonGroup()
            self.desc_group = QtWidgets.QGroupBox("Describe problem with recording")
            self.desc_layout = QtWidgets.QVBoxLayout()
            self.desc_layout.setSpacing(5)
            self.desc_buttons.setExclusive(False)
            options = CriticalInsertionNote.descriptions_gui
            for i, val in enumerate(options):
                button = QtWidgets.QCheckBox(val)
                button.setCheckState(QtCore.Qt.Unchecked)

                self.desc_buttons.addButton(button, id=i)
                self.desc_layout.addWidget(button)

            self.desc_group.setLayout(self.desc_layout)

            self.qc_dialog = QtWidgets.QDialog(self)
            self.qc_dialog.setWindowTitle("QC assessment")
            self.qc_dialog.resize(300, 150)
            self.qc_dialog.accepted.connect(self.qc_button_clicked)
            buttonBox = QtWidgets.QDialogButtonBox(
                QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
            )
            buttonBox.accepted.connect(self.qc_dialog.accept)
            buttonBox.rejected.connect(self.qc_dialog.reject)
            #
            dialog_layout = QtWidgets.QVBoxLayout()
            dialog_layout.addWidget(align_qc_label)
            dialog_layout.addWidget(self.align_qc)
            dialog_layout.addWidget(ephys_qc_label)
            dialog_layout.addWidget(self.ephys_qc)
            dialog_layout.addWidget(self.desc_group)
            dialog_layout.addWidget(buttonBox)
            self.qc_dialog.setLayout(dialog_layout)

    def init_region_lookup(self, allen) -> None:
        """
        Create Allen Atlas structure tree
        """

        # Remove the first row which corresponds to 'Void'
        allen = allen.drop([0]).reset_index(drop=True)

        # Find the parent path of each structure by removing the structure id from path
        def parent_path(struct_path):
            return struct_path.rsplit("/", 2)[0] + "/"

        allen["parent_path"] = allen["structure_id_path"].apply(parent_path)

        # Create standard model view
        self.struct_list = QtGui.QStandardItemModel()
        self.struct_view = QtWidgets.QTreeView()
        self.struct_view.setModel(self.struct_list)
        self.struct_view.clicked.connect(self.label_pressed)

        # Defin
        self.struct_view.setHeaderHidden(True)
        unique_levels = np.unique(allen["depth"]).astype(int)
        parent_info = {}
        idx = np.where(allen["depth"] == unique_levels[0])[0]
        item = QtGui.QStandardItem(
            allen["acronym"][idx[0]] + ": " + allen["name"][idx[0]]
        )
        icon = QtGui.QPixmap(20, 20)
        icon.fill(QtGui.QColor("#" + allen["color_hex_triplet"][idx[0]]))
        item.setIcon(QtGui.QIcon(icon))
        item.setAccessibleText(str(allen["id"][idx[0]]))
        item.setEditable(False)
        self.struct_list.appendRow(item)
        parent_info.update({allen["structure_id_path"][idx[0]]: item})

        for level in unique_levels[1:]:
            idx_levels = np.where(allen["depth"] == level)[0]
            for idx in idx_levels:
                parent = allen["parent_path"][idx]
                parent_item = parent_info[parent]
                item = QtGui.QStandardItem(
                    allen["acronym"][idx] + ": " + allen["name"][idx]
                )
                icon.fill(QtGui.QColor("#" + allen["color_hex_triplet"][idx]))
                item.setIcon(QtGui.QIcon(icon))
                item.setAccessibleText(str(allen["id"][idx]))
                item.setEditable(False)
                parent_item.appendRow(item)
                parent_info.update({allen["structure_id_path"][idx]: item})

        self.struct_description = QtWidgets.QTextEdit()

    def init_figures(self) -> None:
        """
        Create all figures that will be added to the GUI
        """
        # Lists to store the position of probe top and tip
        self.probe_top_lines = []
        self.probe_tip_lines = []

        # Figures to show ephys data
        # 2D scatter/ image plot
        self.fig_img = pg.PlotItem()
        self.fig_img.setYRange(
            min=self.session.probe_tip - self.session.probe_extra,
            max=self.session.probe_top + self.session.probe_extra,
            padding=self.pad,
        )
        self.fig_img.setMouseEnabled(x=False, y=True)
        self.probe_tip_lines.append(
            self.fig_img.addLine(y=self.session.probe_tip, pen=self.kpen_dot, z=50)
        )
        self.probe_top_lines.append(
            self.fig_img.addLine(y=self.session.probe_top, pen=self.kpen_dot, z=50)
        )
        self.set_axis(self.fig_img, "bottom")
        self.fig_data_ax = self.set_axis(
            self.fig_img, "left", label="Distance from probe tip (uV)"
        )

        self.fig_img_cb = pg.PlotItem()
        self.fig_img_cb.setMaximumHeight(70)
        self.fig_img_cb.setMouseEnabled(x=False, y=False)
        self.set_axis(self.fig_img_cb, "bottom", show=False)
        self.set_axis(self.fig_img_cb, "left", pen="w")
        self.set_axis(self.fig_img_cb, "top", pen="w")

        # 1D line plot
        self.fig_line = pg.PlotItem()
        self.fig_line.setMouseEnabled(x=False, y=True)
        self.fig_line.setYRange(
            min=self.session.probe_tip - self.session.probe_extra,
            max=self.session.probe_top + self.session.probe_extra,
            padding=self.pad,
        )
        self.probe_tip_lines.append(
            self.fig_line.addLine(y=self.session.probe_tip, pen=self.kpen_dot, z=50)
        )
        self.probe_top_lines.append(
            self.fig_line.addLine(y=self.session.probe_top, pen=self.kpen_dot, z=50)
        )
        self.set_axis(self.fig_line, "bottom")
        self.set_axis(self.fig_line, "left", show=False)

        # 2D probe plot
        self.fig_probe = pg.PlotItem()
        self.fig_probe.setMouseEnabled(x=False, y=False)
        self.fig_probe.setMaximumWidth(50)
        self.fig_probe.setYRange(
            min=self.session.probe_tip - self.session.probe_extra,
            max=self.session.probe_top + self.session.probe_extra,
            padding=self.pad,
        )
        self.probe_tip_lines.append(
            self.fig_probe.addLine(y=self.session.probe_tip, pen=self.kpen_dot, z=50)
        )
        self.probe_top_lines.append(
            self.fig_probe.addLine(y=self.session.probe_top, pen=self.kpen_dot, z=50)
        )
        self.set_axis(self.fig_probe, "bottom", pen="w")
        self.set_axis(self.fig_probe, "left", show=False)

        self.fig_probe_cb = pg.PlotItem()
        self.fig_probe_cb.setMouseEnabled(x=False, y=False)
        self.fig_probe_cb.setMaximumHeight(70)
        self.set_axis(self.fig_probe_cb, "bottom", show=False)
        self.set_axis(self.fig_probe_cb, "left", show=False)
        self.set_axis(self.fig_probe_cb, "top", pen="w")

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
        self.fig_hist.setYRange(
            min=self.session.probe_tip - self.session.probe_extra,
            max=self.session.probe_top + self.session.probe_extra,
            padding=self.pad,
        )
        self.set_axis(self.fig_hist, "bottom", pen="w")

        self.fig_img.setYLink(self.fig_line)
        self.fig_img.setYLink(self.fig_hist)
        self.fig_line.setYLink(self.fig_hist)
        self.fig_probe.setYLink(self.fig_img)

        replace_axis(self.fig_hist)
        self.ax_hist = self.set_axis(self.fig_hist, "left", pen=None)
        self.ax_hist.setWidth(0)
        # Region labels will be added as TextItems in plot_histology()

        self.fig_scale = pg.PlotItem()
        self.fig_scale.setMaximumWidth(50)
        self.fig_scale.setMouseEnabled(x=False)
        self.scale_label = pg.LabelItem(color="k")
        self.set_axis(self.fig_scale, "bottom", pen="w")
        self.set_axis(self.fig_scale, "left", show=False)
        (self.fig_scale).setYLink(self.fig_hist)

        # Figure that will show scale factor of histology boundaries
        self.fig_scale_cb = pg.PlotItem()
        self.fig_scale_cb.setMouseEnabled(x=False, y=False)
        self.fig_scale_cb.setMaximumHeight(70)
        self.set_axis(self.fig_scale_cb, "bottom", show=False)
        self.set_axis(self.fig_scale_cb, "left", show=False)
        self.fig_scale_ax = self.set_axis(self.fig_scale_cb, "top", pen="w")
        self.set_axis(self.fig_scale_cb, "right", show=False)

        # Histology figure that will remain at initial state for reference
        self.fig_hist_ref = pg.PlotItem()
        self.fig_hist_ref.setMouseEnabled(x=False)
        self.fig_hist_ref.setYRange(
            min=self.session.probe_tip - self.session.probe_extra,
            max=self.session.probe_top + self.session.probe_extra,
            padding=self.pad,
        )
        # Y-link to fig_hist so scrolling/zooming the track-space view stays
        # synchronised with the feature-space plots.
        self.fig_hist_ref.setYLink(self.fig_hist)
        self.set_axis(self.fig_hist_ref, "bottom", pen="w")
        self.set_axis(self.fig_hist_ref, "left", show=False)
        replace_axis(self.fig_hist_ref, orientation="right", pos=(2, 2))
        self.ax_hist_ref = self.set_axis(self.fig_hist_ref, "right", pen=None)
        self.ax_hist_ref.setWidth(0)
        # Region labels will be added as TextItems in plot_histology_ref()

        # Perpendicular histology slice plot
        self.fig_hist_perp = pg.PlotItem()
        self.fig_hist_perp.setMouseEnabled(x=False)
        self.fig_hist_perp.setYRange(
            min=self.session.probe_tip - self.session.probe_extra,
            max=self.session.probe_top + self.session.probe_extra,
            padding=self.pad,
        )
        self.set_axis(self.fig_hist_perp, "left", show=False)
        self.fig_hist_perp.setYLink(self.fig_hist)

        self.fig_hist_area = pg.GraphicsLayoutWidget()
        self.fig_hist_area.setMouseTracking(True)
        self.fig_hist_area.scene().sigMouseClicked.connect(self.on_mouse_double_clicked)
        self.fig_hist_area.scene().sigMouseHover.connect(self.on_mouse_hover)

        self.fig_hist_extra_yaxis = pg.PlotItem()
        self.fig_hist_extra_yaxis.setMouseEnabled(x=False, y=False)
        self.fig_hist_extra_yaxis.setMaximumWidth(2)
        self.fig_hist_extra_yaxis.setYRange(
            min=self.session.probe_tip - self.session.probe_extra,
            max=self.session.probe_top + self.session.probe_extra,
            padding=self.pad,
        )

        self.set_axis(self.fig_hist_extra_yaxis, "bottom", pen="w")
        self.ax_hist2 = self.set_axis(self.fig_hist_extra_yaxis, "left", pen=None)
        self.ax_hist2.setWidth(10)

        self.fig_hist_layout = pg.GraphicsLayout()
        self.fig_hist_layout.addItem(self.fig_scale_cb, 0, 0, 1, 5)  # Span all 5 columns
        self.fig_hist_layout.addItem(self.fig_hist_extra_yaxis, 1, 0)
        self.fig_hist_layout.addItem(self.fig_hist, 1, 1)
        self.fig_hist_layout.addItem(self.fig_hist_perp, 1, 2)  # NEW: Perpendicular slice
        self.fig_hist_layout.addItem(self.fig_scale, 1, 3)        # Moved from column 2
        self.fig_hist_layout.addItem(self.fig_hist_ref, 1, 4)     # Moved from column 3
        self.fig_hist_layout.layout.setColumnStretchFactor(0, 1)
        self.fig_hist_layout.layout.setColumnStretchFactor(1, 4)
        self.fig_hist_layout.layout.setColumnStretchFactor(2, 3)  # Perpendicular slice
        self.fig_hist_layout.layout.setColumnStretchFactor(3, 1)  # Scale
        self.fig_hist_layout.layout.setColumnStretchFactor(4, 4)  # Ref
        self.fig_hist_layout.layout.setRowStretchFactor(0, 1)
        self.fig_hist_layout.layout.setRowStretchFactor(1, 10)
        self.fig_hist_area.addItem(self.fig_hist_layout)

        # Figure to show coronal slice through the brain
        self.fig_slice_area = pg.GraphicsLayoutWidget()
        self.fig_slice_layout = pg.GraphicsLayout()
        self.fig_slice_hist_alt = pg.ViewBox()
        self.fig_slice = pg.ViewBox()
        self.fig_slice_layout.addItem(self.fig_slice, 0, 0)
        self.fig_slice_layout.addItem(self.fig_slice_hist_alt, 0, 1)
        self.fig_slice_layout.layout.setColumnStretchFactor(0, 3)
        self.fig_slice_layout.layout.setColumnStretchFactor(1, 1)
        self.fig_slice_area.addItem(self.fig_slice_layout)
        self.slice_item = self.fig_slice_hist_alt

        # Figure to show fit and offset applied by user
        self.fig_fit = pg.PlotWidget(background="w")
        self.fig_fit.setMouseEnabled(x=False, y=False)
        self.fig_fit_exporter = pg.exporters.ImageExporter(self.fig_fit.plotItem)
        self.fig_fit.sigDeviceRangeChanged.connect(self.on_fig_size_changed)
        self.fig_fit.setXRange(min=self.session.view_total[0], max=self.session.view_total[1])
        self.fig_fit.setYRange(min=self.session.view_total[0], max=self.session.view_total[1])
        # Each point on the fit plot is one user-placed reference line (plus
        # the two implicit endpoints at probe tip / top). X = where the line
        # was placed on the ephys side, Y = where it was placed on the
        # histology side; both are depths along the probe track measured
        # from the first electrode.
        self.set_axis(self.fig_fit, "bottom", label="Ephys reference depth (μm)")
        self.set_axis(self.fig_fit, "left", label="Atlas reference depth (μm)")
        plot = pg.PlotCurveItem()
        plot.setData(x=self.session.depth, y=self.session.depth, pen=self.kpen_dot)
        self.fit_plot = pg.PlotCurveItem(pen=self.bpen_solid)
        self.fit_scatter = pg.ScatterPlotItem(size=7, symbol="o", brush="w", pen="b")
        self.fit_plot_lin = pg.PlotCurveItem(pen=self.rpen_dot)
        self.fig_fit.addItem(plot)
        self.fig_fit.addItem(self.fit_plot)
        self.fig_fit.addItem(self.fit_plot_lin)
        self.fig_fit.addItem(self.fit_scatter)

        self.lin_fit_option = QtWidgets.QCheckBox("Linear fit", self.fig_fit)
        self.lin_fit_option.setChecked(True)
        self.lin_fit_option.stateChanged.connect(self.lin_fit_option_changed)
        self.on_fig_size_changed()

    def on_fig_size_changed(self) -> None:
        # fig_width = self.fig_fit_exporter.getTargetRect().width()
        # fig_height = self.fig_fit_exporter.getTargetRect().width()
        self.lin_fit_option.move(70, 10)


class PopupWindow(QtWidgets.QMainWindow):
    closed = QtCore.pyqtSignal(QtWidgets.QMainWindow)
    moved = QtCore.pyqtSignal()

    def __init__(self, title, parent=None, size=(300, 300), graphics=True) -> None:
        super().__init__()
        self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
        self.resize(size[0], size[1])
        self.move(randrange(30) + 1000, randrange(30) + 200)
        if graphics:
            self.popup_widget = pg.GraphicsLayoutWidget()
        else:
            self.popup_widget = QtWidgets.QWidget()
            self.layout = QtWidgets.QGridLayout()
            self.popup_widget.setLayout(self.layout)
        self.setCentralWidget(self.popup_widget)
        self.setWindowTitle(title)
        self.show()

    def closeEvent(self, event) -> None:
        self.closed.emit(self)
        self.close()

    def leaveEvent(self, event) -> None:
        self.moved.emit()


class CheckableComboBox(QtWidgets.QComboBox):
    def __init__(self) -> None:
        super().__init__()
        self.view().pressed.connect(self.handleItemPressed)
        self.setModel(QtGui.QStandardItemModel(self))

    def handleItemPressed(self, index) -> None:
        item = self.model().itemFromIndex(index)
        if item.checkState() == QtCore.Qt.Checked:
            item.setCheckState(QtCore.Qt.Unchecked)
        else:
            item.setCheckState(QtCore.Qt.Checked)
