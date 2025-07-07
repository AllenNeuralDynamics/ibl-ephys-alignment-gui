import pyqtgraph as pg
from pyqtgraph import debug as debug
from pyqtgraph.functions import makeARGB
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets

import matplotlib

# Originated from
# https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg22889.html
# Modification refered from
# https://gist.github.com/Riateche/27e36977f7d5ea72cf4f

class QRangeSlider(QtWidgets.QSlider):
    sliderMoved = QtCore.pyqtSignal(int, int)
    sliderReleased = QtCore.pyqtSignal(int, int)

    """ A slider for ranges.

        This class provides a dual-slider for ranges, where there is a defined
        maximum and minimum, as is a normal slider, but instead of having a
        single slider value, there are 2 slider values.

        This class emits the same signals as the QSlider base class, with the 
        exception of valueChanged
    """

    def __init__(self, *args):
        super(QRangeSlider, self).__init__(*args)

        self._low = self.minimum()
        self._high = self.maximum()
        self._allow_move = True

        self.pressed_control = QtWidgets.QStyle.SC_None
        self.tick_interval = 0
        self.tick_position = QtWidgets.QSlider.NoTicks
        self.hover_control = QtWidgets.QStyle.SC_None
        self.click_offset = 0

        # 0 for the low, 1 for the high, -1 for both
        self.active_slider = 0

    def low(self):
        return self._low

    def setLow(self, low: int):
        self._low = low
        self.update()

    def high(self):
        return self._high

    def setHigh(self, high):
        self._high = high
        self.update()

    def allowMove(self, allow):
        self._allow_move = allow

    def paintEvent(self, event):
        # based on http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp

        painter = QtGui.QPainter(self)
        style = QtWidgets.QApplication.style()

        # draw groove
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        opt.siderValue = 0
        opt.sliderPosition = 0
        opt.subControls = QtWidgets.QStyle.SC_SliderGroove
        if self.tickPosition() != self.NoTicks:
            opt.subControls |= QtWidgets.QStyle.SC_SliderTickmarks
        style.drawComplexControl(QtWidgets.QStyle.CC_Slider, opt, painter, self)
        groove = style.subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderGroove, self)

        # drawSpan
        # opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        opt.subControls = QtWidgets.QStyle.SC_SliderGroove
        # if self.tickPosition() != self.NoTicks:
        #    opt.subControls |= QtWidgets.QStyle.SC_SliderTickmarks
        opt.siderValue = 0
        # print(self._low)
        opt.sliderPosition = self._low
        low_rect = style.subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderHandle, self)
        opt.sliderPosition = self._high
        high_rect = style.subControlRect(QtWidgets.QStyle.CC_Slider, opt, QtWidgets.QStyle.SC_SliderHandle, self)

        # print(low_rect, high_rect)
        low_pos = self.__pick(low_rect.center())
        high_pos = self.__pick(high_rect.center())

        min_pos = min(low_pos, high_pos)
        max_pos = max(low_pos, high_pos)

        c = QtCore.QRect(low_rect.center(), high_rect.center()).center()
        # print(min_pos, max_pos, c)
        if opt.orientation == QtCore.Qt.Horizontal:
            span_rect = QtCore.QRect(QtCore.QPoint(min_pos, c.y() - 2), QtCore.QPoint(max_pos, c.y() + 1))
        else:
            span_rect = QtCore.QRect(QtCore.QPoint(c.x() - 2, min_pos), QtCore.QPoint(c.x() + 1, max_pos))

        # self.initStyleOption(opt)
        # print(groove.x(), groove.y(), groove.width(), groove.height())
        if opt.orientation == QtCore.Qt.Horizontal:
            groove.adjust(0, 0, -1, 0)
        else:
            groove.adjust(0, 0, 0, -1)

        if True:  # self.isEnabled():
            highlight = self.palette().color(QtGui.QPalette.Highlight)
            painter.setBrush(QtGui.QBrush(highlight))
            painter.setPen(QtGui.QPen(highlight, 0))
            # painter.setPen(QtGui.QPen(self.palette().color(QtGui.QPalette.Dark), 0))
            '''
            if opt.orientation == QtCore.Qt.Horizontal:
                self.setupPainter(painter, opt.orientation, groove.center().x(), groove.top(), groove.center().x(), groove.bottom())
            else:
                self.setupPainter(painter, opt.orientation, groove.left(), groove.center().y(), groove.right(), groove.center().y())
            '''
            # spanRect =
            painter.drawRect(span_rect.intersected(groove))
            # painter.drawRect(groove)

        for i, value in enumerate([self._low, self._high]):
            opt = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(opt)

            # Only draw the groove for the first slider so it doesn't get drawn
            # on top of the existing ones every time
            if i == 0:
                opt.subControls = QtWidgets.QStyle.SC_SliderHandle  # | QtWidgets.QStyle.SC_SliderGroove
            else:
                opt.subControls = QtWidgets.QStyle.SC_SliderHandle

            if self.tickPosition() != self.NoTicks:
                opt.subControls |= QtWidgets.QStyle.SC_SliderTickmarks

            if self.pressed_control:
                opt.activeSubControls = self.pressed_control
            else:
                opt.activeSubControls = self.hover_control

            opt.sliderPosition = value
            opt.sliderValue = value
            style.drawComplexControl(QtWidgets.QStyle.CC_Slider, opt, painter, self)

    def mousePressEvent(self, event):

        if not self._allow_move:
            return

        event.accept()

        style = QtWidgets.QApplication.style()
        button = event.button()

        # In a normal slider control, when the user clicks on a point in the
        # slider's total range, but not on the slider part of the control the
        # control would jump the slider value to where the user clicked.
        # For this control, clicks which are not direct hits will slide both
        # slider parts

        if button:
            opt = QtWidgets.QStyleOptionSlider()
            self.initStyleOption(opt)

            self.active_slider = -1

            for i, value in enumerate([self._low, self._high]):
                opt.sliderPosition = value
                hit = style.hitTestComplexControl(style.CC_Slider, opt, event.pos(), self)
                if hit == style.SC_SliderHandle:
                    self.active_slider = i
                    self.pressed_control = hit

                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)
                    break

            if self.active_slider < 0:
                self.pressed_control = QtWidgets.QStyle.SC_SliderHandle
                self.click_offset = self.__pixelPosToRangeValue(self.__pick(event.pos()))
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self.pressed_control != QtWidgets.QStyle.SC_SliderHandle:
            event.ignore()
            return

        if not self._allow_move:
            return

        event.accept()
        new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)

        if self.active_slider < 0:
            offset = new_pos - self.click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                diff = self.minimum() - self._low
                self._low += diff
                self._high += diff
            if self._high > self.maximum():
                diff = self.maximum() - self._high
                self._low += diff
                self._high += diff
        elif self.active_slider == 0:
            if new_pos >= self._high:
                new_pos = self._high - 1
            self._low = new_pos
        else:
            if new_pos <= self._low:
                new_pos = self._low + 1
            self._high = new_pos

        self.click_offset = new_pos

        self.update()

        # self.emit(QtCore.SIGNAL('sliderMoved(int)'), new_pos)
        self.sliderMoved.emit(self._low, self._high)
        self.sliderReleased.emit(self._low, self._high)

    def __pick(self, pt):
        if self.orientation() == QtCore.Qt.Horizontal:
            return pt.x()
        else:
            return pt.y()

    def __pixelPosToRangeValue(self, pos):
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = QtWidgets.QApplication.style()

        gr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderGroove, self)
        sr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderHandle, self)

        if self.orientation() == QtCore.Qt.Horizontal:
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
        else:
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1

        return style.sliderValueFromPosition(self.minimum(), self.maximum(),
                                             pos - slider_min, slider_max - slider_min,
                                             opt.upsideDown)

class ColorBar(pg.GraphicsWidget):

    def __init__(self, cmap_name, cbin=256, parent=None, data=None):
        pg.GraphicsWidget.__init__(self)

        # Create colour map from matplotlib colourmap name
        self.cmap_name = cmap_name
        cmap = matplotlib.cm.get_cmap(self.cmap_name)
        if type(cmap) == matplotlib.colors.LinearSegmentedColormap:
            cbins = np.linspace(0.0, 1.0, cbin)
            colors = (cmap(cbins)[np.newaxis, :, :3][0]).tolist()
        else:
            colors = cmap.colors
        colors = [(np.array(c) * 255).astype(int).tolist() + [255.] for c in colors]
        positions = np.linspace(0, 1, len(colors))
        self.map = pg.ColorMap(positions, colors)
        self.lut = self.map.getLookupTable()
        self.grad = self.map.getGradient()

    def getBrush(self, data, levels=None):
        if levels is None:
            levels = [np.min(data), np.max(data)]
        brush_rgb, _ = makeARGB(data[:, np.newaxis], levels=levels, lut=self.lut, useRGBA=True)
        brush = [QtGui.QColor(*col) for col in np.squeeze(brush_rgb)]
        return brush

    def getColourMap(self):
        return self.lut

    def makeColourBar(self, width, height, fig, min=0, max=1, label='', lim=False):
        self.cbar = HorizontalBar(width, height, self.grad)
        ax = fig.getAxis('top')
        ax.setPen('k')
        ax.setTextPen('k')
        ax.setStyle(stopAxisAtTick=((True, True)))
        # labelStyle = {'font-size': '8pt'}
        ax.setLabel(label)
        ax.setHeight(30)
        if lim:
            ax.setTicks([[(0, str(np.around(min, 2))), (width, str(np.around(max, 2)))]])
        else:
            ax.setTicks([[(0, str(np.around(min, 2))), (width / 2,
                        str(np.around(min + (max - min) / 2, 2))),
                       (width, str(np.around(max, 2)))],
                [(width / 4, str(np.around(min + (max - min) / 4, 2))),
                 (3 * width / 4, str(np.around(min + (3 * (max - min) / 4), 2)))]])
        fig.setXRange(0, width)
        fig.setYRange(0, height)

        return self.cbar


class HorizontalBar(pg.GraphicsWidget):
    def __init__(self, width, height, grad):
        pg.GraphicsWidget.__init__(self)
        self.width = width
        self.height = height
        self.grad = grad
        QtGui.QPainter()

    def paint(self, p, *args):
        p.setPen(QtCore.Qt.NoPen)
        self.grad.setStart(0, self.height / 2)
        self.grad.setFinalStop(self.width, self.height / 2)
        p.setBrush(pg.QtGui.QBrush(self.grad))
        p.drawRect(QtCore.QRectF(0, 0, self.width, self.height))


class VerticalBar(pg.GraphicsWidget):
    def __init__(self, width, height, grad):
        pg.GraphicsWidget.__init__(self)
        self.width = width
        self.height = height
        self.grad = grad
        QtGui.QPainter()

    def paint(self, p, *args):
        p.setPen(QtCore.Qt.NoPen)
        self.grad.setStart(self.width / 2, self.height)
        self.grad.setFinalStop(self.width / 2, 0)
        p.setBrush(pg.QtGui.QBrush(self.grad))
        p.drawRect(QtCore.QRectF(0, 0, self.width, self.height))


class AdaptedAxisItem(pg.AxisItem):
    def __init__(self, orientation, parent=None,):
        pg.AxisItem.__init__(self, orientation, parent=parent)
        self.style['hideOverlappingLabels'] = False

    def drawPicture(self, p, axisSpec, tickSpecs, textSpecs):
        profiler = debug.Profiler()

        p.setRenderHint(p.Antialiasing, False)
        p.setRenderHint(p.TextAntialiasing, True)

        # draw long line along axis
        pen, p1, p2 = axisSpec
        p.setPen(pen)
        p.drawLine(p1, p2)
        p.translate(0.5, 0)  # resolves some damn pixel ambiguity

        # draw ticks
        for pen, p1, p2 in tickSpecs:
            p.setPen(pen)
            p.drawLine(p1, p2)
        profiler('draw ticks')

        # Draw all text
        if self.style['tickFont'] is not None:
            p.setFont(self.style['tickFont'])
        p.setPen(self.textPen())
        for rect, flags, text in textSpecs:
            p.drawText(rect, int(flags), text)

        profiler('draw text')

    def generateDrawSpecs(self, p):
        """
        Calls tickValues() and tickStrings() to determine where and how ticks should
        be drawn, then generates from this a set of drawing commands to be
        interpreted by drawPicture().
        """
        profiler = debug.Profiler()
        if self.style['tickFont'] is not None:
            p.setFont(self.style['tickFont'])
        bounds = self.mapRectFromParent(self.geometry())

        linkedView = self.linkedView()
        if linkedView is None or self.grid is False:
            tickBounds = bounds
        else:
            tickBounds = linkedView.mapRectToItem(self, linkedView.boundingRect())

        if self.orientation == 'left':
            span = (bounds.topRight(), bounds.bottomRight())
            tickStart = tickBounds.right()
            tickStop = bounds.right()
            tickDir = -1
            axis = 0
        elif self.orientation == 'right':
            span = (bounds.topLeft(), bounds.bottomLeft())
            tickStart = tickBounds.left()
            tickStop = bounds.left()
            tickDir = 1
            axis = 0
        elif self.orientation == 'top':
            span = (bounds.bottomLeft(), bounds.bottomRight())
            tickStart = tickBounds.bottom()
            tickStop = bounds.bottom()
            tickDir = -1
            axis = 1
        elif self.orientation == 'bottom':
            span = (bounds.topLeft(), bounds.topRight())
            tickStart = tickBounds.top()
            tickStop = bounds.top()
            tickDir = 1
            axis = 1
        else:
            raise ValueError("self.orientation must be in ('left', 'right', 'top', 'bottom')")
        # print tickStart, tickStop, span

        ## determine size of this item in pixels
        points = list(map(self.mapToDevice, span))
        if None in points:
            return
        lengthInPixels = pg.Point(points[1] - points[0]).length()
        if lengthInPixels == 0:
            return

        # Determine major / minor / subminor axis ticks
        if self._tickLevels is None:
            tickLevels = self.tickValues(self.range[0], self.range[1], lengthInPixels)
            tickStrings = None
        else:
            ## parse self.tickLevels into the formats returned by tickLevels() and tickStrings()
            tickLevels = []
            tickStrings = []
            for level in self._tickLevels:
                values = []
                strings = []
                tickLevels.append((None, values))
                tickStrings.append(strings)
                for val, strn in level:
                    values.append(val)
                    strings.append(strn)

        ## determine mapping between tick values and local coordinates
        dif = self.range[1] - self.range[0]
        if dif == 0:
            xScale = 1
            offset = 0
        else:
            if axis == 0:
                xScale = -bounds.height() / dif
                offset = self.range[0] * xScale - bounds.height()
            else:
                xScale = bounds.width() / dif
                offset = self.range[0] * xScale

        xRange = [x * xScale - offset for x in self.range]
        xMin = min(xRange)
        xMax = max(xRange)

        profiler('init')

        tickPositions = []  # remembers positions of previously drawn ticks

        ## compute coordinates to draw ticks
        ## draw three different intervals, long ticks first
        tickSpecs = []
        for i in range(len(tickLevels)):
            tickPositions.append([])
            ticks = tickLevels[i][1]

            ## length of tick
            tickLength = self.style['tickLength'] / ((i * 0.5) + 1.0)

            lineAlpha = self.style["tickAlpha"]
            if lineAlpha is None:
                lineAlpha = 255 / (i + 1)
                if self.grid is not False:
                    lineAlpha *= self.grid / 255. * pg.functions.clip_scalar((0.05 * lengthInPixels / (len(ticks) + 1)), 0., 1.)
            elif isinstance(lineAlpha, float):
                lineAlpha *= 255
                lineAlpha = max(0, int(round(lineAlpha)))
                lineAlpha = min(255, int(round(lineAlpha)))
            elif isinstance(lineAlpha, int):
                if (lineAlpha > 255) or (lineAlpha < 0):
                    raise ValueError("lineAlpha should be [0..255]")
            else:
                raise TypeError("Line Alpha should be of type None, float or int")

            for v in ticks:
                # determine actual position to draw this tick
                x = (v * xScale) - offset
                if x < xMin or x > xMax:  # last check to make sure no out-of-bounds ticks are drawn
                    tickPositions[i].append(None)
                    continue
                tickPositions[i].append(x)

                p1 = [x, x]
                p2 = [x, x]
                p1[axis] = tickStart
                p2[axis] = tickStop
                if self.grid is False:
                    p2[axis] += tickLength * tickDir
                tickPen = self.pen()
                color = tickPen.color()
                color.setAlpha(int(lineAlpha))
                tickPen.setColor(color)
                tickSpecs.append((tickPen, pg.Point(p1), pg.Point(p2)))
        profiler('compute ticks')

        if self.style['stopAxisAtTick'][0] is True:
            minTickPosition = min(map(min, tickPositions))
            if axis == 0:
                stop = max(span[0].y(), minTickPosition)
                span[0].setY(stop)
            else:
                stop = max(span[0].x(), minTickPosition)
                span[0].setX(stop)
        if self.style['stopAxisAtTick'][1] is True:
            maxTickPosition = max(map(max, tickPositions))
            if axis == 0:
                stop = min(span[1].y(), maxTickPosition)
                span[1].setY(stop)
            else:
                stop = min(span[1].x(), maxTickPosition)
                span[1].setX(stop)
        axisSpec = (self.pen(), span[0], span[1])

        textOffset = self.style['tickTextOffset'][axis]  # spacing between axis and text
        # if self.style['autoExpandTextSpace'] is True:
        # textWidth = self.textWidth
        # textHeight = self.textHeight
        # else:
        # textWidth = self.style['tickTextWidth'] ## space allocated for horizontal text
        # textHeight = self.style['tickTextHeight'] ## space allocated for horizontal text

        textSize2 = 0
        lastTextSize2 = 0
        textRects = []
        textSpecs = []  # list of draw

        # If values are hidden, return early
        if not self.style['showValues']:
            return (axisSpec, tickSpecs, textSpecs)

        for i in range(min(len(tickLevels), self.style['maxTextLevel'] + 1)):
            ## Get the list of strings to display for this level
            if tickStrings is None:
                spacing, values = tickLevels[i]
                strings = self.tickStrings(values, self.autoSIPrefixScale * self.scale, spacing)
            else:
                strings = tickStrings[i]

            if len(strings) == 0:
                continue

            ## ignore strings belonging to ticks that were previously ignored
            for j in range(len(strings)):
                if tickPositions[i][j] is None:
                    strings[j] = None

            ## Measure density of text; decide whether to draw this level
            rects = []
            for s in strings:
                if s is None:
                    rects.append(None)
                else:
                    br = p.boundingRect(pg.QtCore.QRectF(0, 0, 100, 100), pg.QtCore.Qt.AlignmentFlag.AlignCenter, s)
                    ## boundingRect is usually just a bit too large
                    ## (but this probably depends on per-font metrics?)
                    br.setHeight(br.height() * 0.8)

                    rects.append(br)
                    textRects.append(rects[-1])

            if len(textRects) > 0:
                ## measure all text, make sure there's enough room
                if axis == 0:
                    textSize = np.sum([r.height() for r in textRects])
                    textSize2 = np.max([r.width() for r in textRects])
                else:
                    textSize = np.sum([r.width() for r in textRects])
                    textSize2 = np.max([r.height() for r in textRects])
            else:
                textSize = 0
                textSize2 = 0

            if i > 0:  # always draw top level
                # If the strings are too crowded, stop drawing text now.
                # We use three different crowding limits based on the number
                # of texts drawn so far.
                textFillRatio = float(textSize) / lengthInPixels
                finished = False
                for nTexts, limit in self.style['textFillLimits']:
                    if len(textSpecs) >= nTexts and textFillRatio >= limit:
                        finished = True
                        break
                if finished:
                    break

            lastTextSize2 = textSize2

            # spacing, values = tickLevels[best]
            # strings = self.tickStrings(values, self.scale, spacing)
            # Determine exactly where tick text should be drawn
            for j in range(len(strings)):
                vstr = strings[j]
                if vstr is None:  # this tick was ignored because it is out of bounds
                    continue
                x = tickPositions[i][j]
                # textRect = p.boundingRect(QtCore.QRectF(0, 0, 100, 100), QtCore.Qt.AlignmentFlag.AlignCenter, vstr)
                textRect = rects[j]
                height = textRect.height()
                width = textRect.width()
                # self.textHeight = height
                offset = max(0, self.style['tickLength']) + textOffset

                rect = pg.QtCore.QRectF()
                if self.orientation == 'left':
                    alignFlags = pg.QtCore.Qt.AlignmentFlag.AlignRight | pg.QtCore.Qt.AlignmentFlag.AlignVCenter
                    rect = pg.QtCore.QRectF(tickStop - offset - width, x - (height / 2), width, height)
                elif self.orientation == 'right':
                    alignFlags = pg.QtCore.Qt.AlignmentFlag.AlignLeft | pg.QtCore.Qt.AlignmentFlag.AlignVCenter
                    rect = pg.QtCore.QRectF(tickStop + offset, x - (height / 2), width, height)
                elif self.orientation == 'top':
                    alignFlags = pg.QtCore.Qt.AlignmentFlag.AlignHCenter | pg.QtCore.Qt.AlignmentFlag.AlignBottom
                    rect = pg.QtCore.QRectF(x - width / 2., tickStop - offset - height, width, height)
                elif self.orientation == 'bottom':
                    alignFlags = pg.QtCore.Qt.AlignmentFlag.AlignHCenter | pg.QtCore.Qt.AlignmentFlag.AlignTop
                    rect = pg.QtCore.QRectF(x - width / 2., tickStop + offset, width, height)

                textFlags = alignFlags | pg.QtCore.Qt.TextFlag.TextDontClip
                # p.setPen(self.pen())
                # p.drawText(rect, textFlags, vstr)

                # br = self.boundingRect()
                # if not br.contains(rect):
                #     continue

                textSpecs.append((rect, textFlags, vstr))
        profiler('compute text')

        ## update max text size if needed.
        self._updateMaxTextSize(lastTextSize2)

        return (axisSpec, tickSpecs, textSpecs)


def replace_axis(plot_item, orientation='left', pos=(2, 0)):

    new_axis = AdaptedAxisItem(orientation, parent=plot_item)
    oldAxis = plot_item.axes[orientation]['item']
    plot_item.layout.removeItem(oldAxis)
    oldAxis.unlinkFromView()
    #
    new_axis.linkToView(plot_item.vb)
    plot_item.axes[orientation] = {'item': new_axis, 'pos': pos}
    plot_item.layout.addItem(new_axis, *pos)
    new_axis.setZValue(-1000)
    new_axis.setFlag(new_axis.ItemNegativeZStacksBehindParent)
