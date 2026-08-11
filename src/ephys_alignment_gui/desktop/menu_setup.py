"""Desktop menu construction."""

from __future__ import annotations

from typing import Any

from PyQt5 import QtWidgets


def build_menu_bar(window: Any) -> None:
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
    menu_bar = QtWidgets.QMenuBar(window)
    menu_bar.setNativeMenuBar(False)
    window.setMenuBar(menu_bar)

    window.displays.ephys.attach_plot_menus(menu_bar)

    window.displays.slice.attach_slice_menu(
        menu_bar,
        parent=window,
        offline=window.offline,
    )

    window.displays.ephys.attach_unit_filter_menu(menu_bar, window)

    # FIT OPTIONS MENU BAR
    # Define all possible keyboard shortcut interactions for GUI

    # Shortcut to apply interpolation
    fit_option = QtWidgets.QAction("Fit", window)
    fit_option.setShortcut("Return")
    fit_option.triggered.connect(window.fit_button_pressed)

    # Shortcuts to apply offset
    offset_option = QtWidgets.QAction("Offset", window)
    offset_option.setShortcut("O")
    offset_option.triggered.connect(window.offset_button_pressed)
    moveup_option = QtWidgets.QAction("Offset + 50um", window)
    moveup_option.setShortcut("Shift+Up")
    moveup_option.triggered.connect(window.moveup_button_pressed)
    movedown_option = QtWidgets.QAction("Offset - 50um", window)
    movedown_option.setShortcut("Shift+Down")
    movedown_option.triggered.connect(window.movedown_button_pressed)

    # Shortcut to delete a reference line
    delete_line_option = QtWidgets.QAction("Delete Line", window)
    delete_line_option.setShortcut("Shift+D")
    delete_line_option.triggered.connect(window.delete_line_button_pressed)

    # Shortcut to move between previous/next moves
    next_option = QtWidgets.QAction("Next", window)
    next_option.setShortcut("Right")
    next_option.triggered.connect(window.next_button_pressed)
    prev_option = QtWidgets.QAction("Previous", window)
    prev_option.setShortcut("Left")
    prev_option.triggered.connect(window.prev_button_pressed)

    # Shortcut to reset GUI to initial state
    reset_option = QtWidgets.QAction("Reset", window)
    reset_option.setShortcut("Ctrl+R")
    reset_option.triggered.connect(window.reset_button_pressed)

    # Shortcut to save final state to JSON file
    complete_option = QtWidgets.QAction("Save", window)
    complete_option.setShortcut("Ctrl+S")
    if not window.offline:
        complete_option.triggered.connect(window.display_qc_options)
    else:
        complete_option.triggered.connect(window.complete_button_pressed_offline)

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
    toggle1_option = QtWidgets.QAction("Toggle Image Plots", window)
    toggle1_option.setShortcut("Alt+1")
    toggle1_option.triggered.connect(lambda: window.displays.ephys.toggle_plot("image"))
    toggle2_option = QtWidgets.QAction("Toggle Line Plots", window)
    toggle2_option.setShortcut("Alt+2")
    toggle2_option.triggered.connect(lambda: window.displays.ephys.toggle_plot("line"))
    toggle3_option = QtWidgets.QAction("Toggle Probe Plots", window)
    toggle3_option.setShortcut("Alt+3")
    toggle3_option.triggered.connect(lambda: window.displays.ephys.toggle_plot("probe"))
    toggle4_option = QtWidgets.QAction("Toggle Slice Plots", window)
    toggle4_option.setShortcut("Alt+4")
    toggle4_option.triggered.connect(lambda: window.displays.slice.toggle_slice_plot())

    toggle5_option = QtWidgets.QAction("Toggle Previous Image Plots", window)
    toggle5_option.setShortcut("Alt+Ctrl+1")
    toggle5_option.triggered.connect(
        lambda: window.displays.ephys.toggle_plot("image", reverse=True)
    )
    toggle6_option = QtWidgets.QAction("Toggle Previous Line Plots", window)
    toggle6_option.setShortcut("Alt+Ctrl+2")
    toggle6_option.triggered.connect(
        lambda: window.displays.ephys.toggle_plot("line", reverse=True)
    )
    toggle7_option = QtWidgets.QAction("Toggle Previous Probe Plots", window)
    toggle7_option.setShortcut("Alt+Ctrl+3")
    toggle7_option.triggered.connect(
        lambda: window.displays.ephys.toggle_plot("probe", reverse=True)
    )
    toggle8_option = QtWidgets.QAction("Toggle Previous Slice Plots", window)
    toggle8_option.setShortcut("Alt+Ctrl+4")
    toggle8_option.triggered.connect(
        lambda: window.displays.slice.toggle_slice_plot(reverse=True)
    )

    # Shortcuts to switch order of 3 panels in ephys plot
    view1_option = QtWidgets.QAction("View 1", window)
    view1_option.setShortcut("Shift+1")
    view1_option.triggered.connect(lambda: window.set_view(view=1))
    view2_option = QtWidgets.QAction("View 2", window)
    view2_option.setShortcut("Shift+2")
    view2_option.triggered.connect(lambda: window.set_view(view=2))
    view3_option = QtWidgets.QAction("View 3", window)
    view3_option.setShortcut("Shift+3")
    view3_option.triggered.connect(lambda: window.set_view(view=3))

    # Shortcut to reset axis on figures
    axis_option = QtWidgets.QAction("Reset Axis", window)
    axis_option.setShortcut("Shift+A")
    axis_option.triggered.connect(window.reset_axis_button_pressed)

    # Shortcut to hide/show region labels
    toggle_labels_option = QtWidgets.QAction("Hide/Show Labels", window)
    toggle_labels_option.setShortcut("Shift+L")
    toggle_labels_option.triggered.connect(window.toggle_labels_button_pressed)

    # Shortcut to hide/show reference lines
    toggle_lines_option = QtWidgets.QAction("Hide/Show Lines", window)
    toggle_lines_option.setShortcut("Shift+H")
    toggle_lines_option.triggered.connect(window.toggle_line_button_pressed)

    # Shortcut to hide/show reference lines and channels on slice image
    toggle_channels_option = QtWidgets.QAction("Hide/Show Channels", window)
    toggle_channels_option.setShortcut("Shift+C")
    toggle_channels_option.triggered.connect(window.toggle_channel_button_pressed)

    # Shortcut to change default histology reference image
    toggle_histology_option = QtWidgets.QAction("Hide/Show Nearby Boundaries", window)
    toggle_histology_option.setShortcut("Shift+N")
    toggle_histology_option.triggered.connect(window.toggle_histology_button_pressed)

    # Option to change histology regions from Allen to Franklin Paxinos
    toggle_histology_map_option = QtWidgets.QAction("Change Histology Map", window)
    toggle_histology_map_option.setShortcut("Shift+M")
    toggle_histology_map_option.triggered.connect(
        window.toggle_histology_map_button_pressed
    )

    # Shortcuts for cluster popup window
    popup_minimise = QtWidgets.QAction("Minimise/Show Cluster Popup", window)
    popup_minimise.setShortcut("Alt+M")
    popup_minimise.triggered.connect(window.minimise_popups)
    popup_close = QtWidgets.QAction("Close Cluster Popup", window)
    popup_close.setShortcut("Alt+X")
    popup_close.triggered.connect(window.close_popups)

    # Option to save all plots
    save_plots = QtWidgets.QAction("Save Plots", window)
    save_plots.setShortcut("Ctrl+Shift+S")
    save_plots.triggered.connect(window.save_plots)

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
    session_notes = QtWidgets.QAction("Session Notes", window)
    session_notes.triggered.connect(window.display_session_notes)
    # Shortcut to show label information
    region_info = QtWidgets.QAction("Region Info", window)
    region_info.setShortcut("Shift+I")
    region_info.triggered.connect(window.describe_labels_pressed)

    # Add menu bar with all possible session info options
    info_options = menu_bar.addMenu("Session Information")
    info_options.addAction(session_notes)
    info_options.addAction(region_info)

    # Display other sessions that are closeby if online mode
    if not window.offline:
        nearby_info = QtWidgets.QAction("Nearby Sessions", window)
        nearby_info.triggered.connect(window.display_nearby_sessions)
        info_options.addAction(nearby_info)
