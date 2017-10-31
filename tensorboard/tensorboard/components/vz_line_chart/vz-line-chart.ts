/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
/* tslint:disable:no-namespace variable-name */

import {DragZoomLayer} from './dragZoomInteraction.js';
import * as ChartHelpers from './vz-chart-helpers.js';

Polymer({
  is: 'vz-line-chart',
  properties: {
    /**
     * Scale that maps series names to colors. The default colors are from
     * d3.schemeCategory10. Use this property to replace the default line
     * colors with colors of your own choice.
     * @type {Plottable.Scales.Color}
     * @required
     */
    colorScale: {
      type: Object,
      value: function() {
        return new Plottable.Scales.Color().range(d3.schemeCategory10);
      }
    },

    /**
     * Whether smoothing is enabled or not. If true, smoothed lines will be
     * plotted in the chart while the unsmoothed lines will be ghosted in
     * the background.
     *
     * The smoothing algorithm is a simple moving average, which, given a
     * point p and a window w, replaces p with a simple average of the
     * points in the [p - floor(w/2), p + floor(w/2)] range.  If there
     * aren't enough points to cover the entire window to the left, the
     * window is reduced to fit exactly the amount of elements available.
     * This means that the smoothed line will be less in and gradually
     * become more smooth until the desired window is reached. However when
     * there aren't enough points on the right, the line stops being
     * rendered at all.
     */
    smoothingEnabled: {
      type: Boolean,
      notify: true,
      value: false,
    },

    /**
     * Weight (between 0.0 and 1.0) of the smoothing. This weight controls
     * the window size, and a weight of 1.0 means using 50% of the entire
     * dataset as the window, while a weight of 0.0 means using a window of
     * 0 (and thus replacing each point with themselves).
     *
     * The growth between 0.0 and 1.0 is not linear though. Because
     * changing the window from 0% to 30% of the dataset smooths the line a
     * lot more than changing the window from 70% to 100%, an exponential
     * function is used instead: http://i.imgur.com/bDrhEZU.png. This
     * function increases the size of the window slowly at the beginning
     * and gradually speeds up the growth, but 0.0 still means a window of
     * 0 and 1.0 still means a window of the dataset's length.
     */
    smoothingWeight: {type: Number, value: 0.6},

    /**
     * We accept a function for creating an XComponents object instead of such
     * an object itself because the Axis must be made right when we make the
     * LineChart object, lest we use a previously destroyed Axis. See the async
     * logic below that uses this property.
     *
     * Note that this function returns a function because polymer calls the
     * outer function to compute the value. We actually want the value of this
     * property to be the inner function.
     *
     * @type {function(): ChartHelpers.XComponents}
     */
    xComponentsCreationMethod: {type: Object, value: () => ChartHelpers.stepX},

    /**
     * A method that implements the Plottable.IAccessor<number> interface. Used
     * for accessing the y value from a data point.
     *
     * Note that this function returns a function because polymer calls the
     * outer function to compute the value. We actually want the value of this
     * property to be the inner function.
     */
    yValueAccessor: {type: Object, value: () => (d => d.scalar)},

    /**
     * An array of ChartHelper.TooltipColumn objects. Used to populate the table
     * within the tooltip. The table contains 1 row per run.
     *
     * Note that this function returns a function because polymer calls the
     * outer function to compute the value. We actually want the value of this
     * property to be the inner function.
     *
     */
    tooltipColumns: {
      type: Array,
      value: function() {
        const valueFormatter = ChartHelpers.multiscaleFormatter(
            ChartHelpers.Y_TOOLTIP_FORMATTER_PRECISION);
        const formatValueOrNaN = (x) => isNaN(x) ? 'NaN' : valueFormatter(x);

        return [
          {
            title: 'Name',
            evaluate: (d) => d.dataset.metadata().name,
          },
          {
            title: 'Smoothed',
            evaluate: (d, statusObject) => formatValueOrNaN(
                statusObject.smoothingEnabled ? d.datum.smoothed :
                                                d.datum.scalar),
          },
          {
            title: 'Value',
            evaluate: (d) => formatValueOrNaN(d.datum.scalar),
          },
          {
            title: 'Step',
            evaluate: (d) => ChartHelpers.stepFormatter(d.datum.step),
          },
          {
            title: 'Time',
            evaluate: (d) => ChartHelpers.timeFormatter(d.datum.wall_time),
          },
          {
            title: 'Relative',
            evaluate: (d) => ChartHelpers.relativeFormatter(
                ChartHelpers.relativeAccessor(d.datum, -1, d.dataset)),
          },
        ];
      }
    },

    /**
     * An optional array of 2 numbers for the min and max of the default range
     * of the Y axis. If not provided, a reasonable range will be generated.
     * This property is a list instead of 2 individual properties to emphasize
     * that both the min and the max must be specified (or neither at all).
     */
    defaultXRange: Array,

    /**
     * An optional array of 2 numbers for the min and max of the default range
     * of the Y axis. If not provided, a reasonable range will be generated.
     * This property is a list instead of 2 individual properties to emphasize
     * that both the min and the max must be specified (or neither at all).
     */
    defaultYRange: Array,

    /**
     * Tooltip header innerHTML text. We cannot use a dom-repeat inside of a
     * table element because Polymer does not support that. This seems like
     * a bug in Polymer. Hence, we manually generate the HTML for creating a row
     * of table headers.
     */
    _tooltipTableHeaderHtml: {
      type: String,
      computed: "_computeTooltipTableHeaderHtml(tooltipColumns)",
    },

    /**
     * The scale for the y-axis. Allows:
     * - "linear" - linear scale (Plottable.Scales.Linear)
     * - "log" - modified-log scale (Plottable.Scales.ModifiedLog)
     */
    yScaleType: {type: String, value: 'linear'},

    /**
     * Whether to ignore outlier data when computing the yScale domain.
     */

    ignoreYOutliers: {
      type: Boolean,
      value: false,
    },

    /**
     * Change how the tooltip is sorted. Allows:
     * - "default" - Sort the tooltip by input order.
     * - "ascending" - Sort the tooltip by ascending value.
     * - "descending" - Sort the tooltip by descending value.
     * - "nearest" - Sort the tooltip by closest to cursor.
     */
    tooltipSortingMethod: {type: String, value: 'default'},

    /**
     * Change how the tooltip is positioned. Allows:
     * - "bottom" - Position the tooltip on the bottom of the chart.
     * - "right" - Position the tooltip to the right of the chart.
     */
    tooltipPosition: {type: String, value: 'bottom'},

    _attached: Boolean,
    _chart: Object,
    _visibleSeriesCache: {
      type: Array,
      value: function() {
        return []
      }
    },
    _seriesDataCache: {
      type: Object,
      value: function() {
        return {}
      }
    },
    _makeChartAsyncCallbackId: {type: Number, value: null}
  },
  observers: [
    '_makeChart(xComponentsCreationMethod, yValueAccessor, yScaleType, tooltipColumns, colorScale, _attached)',
    '_reloadFromCache(_chart)',
    '_smoothingChanged(smoothingEnabled, smoothingWeight, _chart)',
    '_tooltipSortingMethodChanged(tooltipSortingMethod, _chart)',
    '_tooltipPositionChanged(tooltipPosition, _chart)',
    '_outliersChanged(ignoreYOutliers, _chart)'
  ],

  /**
   * Sets the series that the chart displays. Series with other names will
   * not be displayed.
   *
   * @param {Array<String>} names Array with the names of the series to
   * display.
   */
  setVisibleSeries: function(names) {
    this._visibleSeriesCache = names;
    if (this._chart) {
      this._chart.setVisibleSeries(names);
      this.redraw();
    }
  },

  /**
   * Sets the data of one of the series. Note that to display this series
   * its name must be in the setVisibleSeries() array.
   *
   * @param {string} name Name of the series.
   * @param {Array<ChartHelpers.ScalarDatum>} data Data of the series. This is
   * an array of objects with at least the following properties:
   * - step: (Number) - index of the datum.
   * - wall_time: (Date) - Date object with the datum's time.
   * - scalar: (Number) - Value of the datum.
   */
  setSeriesData: function(name, data) {
    this._seriesDataCache[name] = data;
    if (this._chart) {
      this._chart.setSeriesData(name, data);
    }
  },

  /**
   * Reset the chart domain. If the chart has not rendered yet, a call to this
   * method no-ops.
   */
  resetDomain: function() {
    if (this._chart) {
      this._chart.resetDomain();
    }
  },

  /**
   * Re-renders the chart. Useful if e.g. the container size changed.
   */
  redraw: function() {
    if (this._chart) {
      this._chart.redraw();
    }
  },
  attached: function() {
    this._attached = true;
  },
  detached: function() {
    this._attached = false;
  },
  ready: function() {
    this.scopeSubtree(this.$.tooltip, true);
    this.scopeSubtree(this.$.chartdiv, true);
  },

  /**
   * Creates a chart, and asynchronously renders it. Fires a chart-rendered
   * event after the chart is rendered.
   */
  _makeChart: function(
      xComponentsCreationMethod,
      yValueAccessor,
      yScaleType,
      tooltipColumns,
      colorScale,
      _attached) {
    if (this._makeChartAsyncCallbackId !== null) {
      this.cancelAsync(this._makeChartAsyncCallbackId);
      this._makeChartAsyncCallbackId = null;
    }

    this._makeChartAsyncCallbackId = this.async(function() {
      this._makeChartAsyncCallbackId = null;
      if (!this._attached ||
          !this.xComponentsCreationMethod ||
          !this.yValueAccessor ||
          !this.tooltipColumns) {
        return;
      }
      if (this._chart) this._chart.destroy();
      var tooltip = d3.select(this.$.tooltip);
      // We directly reference properties of `this` because this call is
      // asynchronous, and values may have changed in between the call being
      // initiated and actually being run.
      var chart = new LineChart(
          this.xComponentsCreationMethod,
          this.yValueAccessor,
          yScaleType,
          colorScale,
          tooltip,
          this.tooltipColumns,
          this.defaultXRange,
          this.defaultYRange);
      var div = d3.select(this.$.chartdiv);
      chart.renderTo(div);
      this._chart = chart;
    }, 350);
  },
  _reloadFromCache: function() {
    if (this._chart) {
      this._chart.setVisibleSeries(this._visibleSeriesCache);
      this._visibleSeriesCache.forEach(function(name) {
        this._chart.setSeriesData(name, this._seriesDataCache[name] || []);
      }.bind(this));
    }
  },
  _smoothingChanged: function() {
    if (!this._chart) {
      return;
    }
    if (this.smoothingEnabled) {
      this._chart.smoothingUpdate(this.smoothingWeight);
    } else {
      this._chart.smoothingDisable();
    }
  },
  _outliersChanged: function() {
    if (!this._chart) {
      return;
    }
    this._chart.ignoreYOutliers(this.ignoreYOutliers);
  },
  _tooltipSortingMethodChanged: function() {
    if (this._chart) {
      this._chart.setTooltipSortingMethod(this.tooltipSortingMethod);
    }
  },
  _tooltipPositionChanged: function() {
    if (this._chart) {
      this._chart.setTooltipPosition(this.tooltipPosition);
    }
  },
  _computeTooltipTableHeaderHtml(tooltipColumns) {
    // The first column contains the circle with the color of the run.
    const titles = ["", ..._.map(tooltipColumns, 'title')];
    return titles.map(title => `<th>${this._sanitize(title)}</th>`).join('');
  },
  _sanitize(value) {
    return value.replace(/</g, '&lt;')
                .replace(/>/g, '&gt;')  // for symmetry :-)
                .replace(/&/g, '&amp;');
  },
});

class LineChart {
  private name2datasets: {[name: string]: Plottable.Dataset};
  private seriesNames: string[];

  private xAccessor: Plottable.IAccessor<number|Date>;
  private xScale: Plottable.QuantitativeScale<number|Date>;
  private yScale: Plottable.QuantitativeScale<number>;
  private gridlines: Plottable.Components.Gridlines;
  private center: Plottable.Components.Group;
  private xAxis: Plottable.Axes.Numeric|Plottable.Axes.Time;
  private yAxis: Plottable.Axes.Numeric;
  private outer: Plottable.Components.Table;
  private colorScale: Plottable.Scales.Color;
  private tooltip: d3.Selection<any, any, any, any>;
  private dzl: DragZoomLayer;

  private linePlot: Plottable.Plots.Line<number|Date>;
  private smoothLinePlot: Plottable.Plots.Line<number|Date>;
  private scatterPlot: Plottable.Plots.Scatter<number|Date, Number>;
  private nanDisplay: Plottable.Plots.Scatter<number|Date, Number>;
  private yValueAccessor: Plottable.IAccessor<number>;
  private smoothedAccessor: Plottable.IAccessor<number>;
  private lastPointsDataset: Plottable.Dataset;
  private datasets: Plottable.Dataset[];
  private onDatasetChanged: (dataset: Plottable.Dataset) => void;
  private nanDataset: Plottable.Dataset;
  private smoothingWeight: number;
  private smoothingEnabled: boolean;
  private tooltipSortingMethod: string;
  private tooltipPosition: string;
  private _ignoreYOutliers: boolean;

  // An optional list of 2 numbers.
  private _defaultXRange: number[];
  // An optional list of 2 numbers.
  private _defaultYRange: number[];

  private targetSVG: d3.Selection<any, any, any, any>;

  constructor(
      xComponentsCreationMethod: () => ChartHelpers.XComponents,
      yValueAccessor: Plottable.IAccessor<number>,
      yScaleType: string,
      colorScale: Plottable.Scales.Color,
      tooltip: d3.Selection<any, any, any, any>,
      tooltipColumns: ChartHelpers.TooltipColumn[],
      defaultXRange?: number[],
      defaultYRange?: number[]) {
    this.seriesNames = [];
    this.name2datasets = {};
    this.colorScale = colorScale;
    this.tooltip = tooltip;
    this.datasets = [];
    this._ignoreYOutliers = false;
    // lastPointDataset is a dataset that contains just the last point of
    // every dataset we're currently drawing.
    this.lastPointsDataset = new Plottable.Dataset();
    this.nanDataset = new Plottable.Dataset();
    this.yValueAccessor = yValueAccessor;
    // need to do a single bind, so we can deregister the callback from
    // old Plottable.Datasets. (Deregistration is done by identity checks.)
    this.onDatasetChanged = this._onDatasetChanged.bind(this);

    this._defaultXRange = defaultXRange;
    this._defaultYRange = defaultYRange;

    this.buildChart(
        xComponentsCreationMethod, yValueAccessor, yScaleType, tooltipColumns);
  }

  private buildChart(
      xComponentsCreationMethod: () => ChartHelpers.XComponents,
      yValueAccessor: Plottable.IAccessor<number>,
      yScaleType: string,
      tooltipColumns: ChartHelpers.TooltipColumn[]) {
    if (this.outer) {
      this.outer.destroy();
    }
    const xComponents = xComponentsCreationMethod();
    this.xAccessor = xComponents.accessor;
    this.xScale = xComponents.scale;
    this.xAxis = xComponents.axis;
    this.xAxis.margin(0).tickLabelPadding(3);
    this.yScale = LineChart.getYScaleFromType(yScaleType);
    this.yAxis = new Plottable.Axes.Numeric(this.yScale, 'left');
    let yFormatter = ChartHelpers.multiscaleFormatter(
        ChartHelpers.Y_AXIS_FORMATTER_PRECISION);
    this.yAxis.margin(0).tickLabelPadding(5).formatter(yFormatter);
    this.yAxis.usesTextWidthApproximation(true);

    this.dzl = new DragZoomLayer(
        this.xScale, this.yScale, this.resetYDomain.bind(this));

    let center = this.buildPlot(
        this.xScale,
        this.yScale,
        tooltipColumns);

    this.gridlines =
        new Plottable.Components.Gridlines(this.xScale, this.yScale);

    let xZeroLine = new Plottable.Components.GuideLineLayer('horizontal');
    xZeroLine.scale(this.yScale).value(0);
    let yZeroLine = new Plottable.Components.GuideLineLayer('vertical');
    yZeroLine.scale(this.xScale).value(0);

    this.center = new Plottable.Components.Group(
        [this.gridlines, xZeroLine, yZeroLine, center, this.dzl]);
    this.outer = new Plottable.Components.Table(
        [[this.yAxis, this.center], [null, this.xAxis]]);
  }

  private buildPlot(xScale, yScale, tooltipColumns): Plottable.Component {
    this.smoothedAccessor = (d: ChartHelpers.ScalarDatum) => d.smoothed;
    let linePlot = new Plottable.Plots.Line<number|Date>();
    linePlot.x(this.xAccessor, xScale);
    linePlot.y(this.yValueAccessor, yScale);
    linePlot.attr(
        'stroke',
        (d: ChartHelpers.Datum, i: number, dataset: Plottable.Dataset) =>
            this.colorScale.scale(dataset.metadata().name));
    this.linePlot = linePlot;
    const group = this.setupTooltips(linePlot, tooltipColumns);

    let smoothLinePlot = new Plottable.Plots.Line<number|Date>();
    smoothLinePlot.x(this.xAccessor, xScale);
    smoothLinePlot.y(this.smoothedAccessor, yScale);
    smoothLinePlot.attr(
        'stroke',
        (d: ChartHelpers.Datum, i: number, dataset: Plottable.Dataset) =>
            this.colorScale.scale(dataset.metadata().name));
    this.smoothLinePlot = smoothLinePlot;

    // The scatterPlot will display the last point for each dataset.
    // This way, if there is only one datum for the series, it is still
    // visible. We hide it when tooltips are active to keep things clean.
    let scatterPlot = new Plottable.Plots.Scatter<number|Date, number>();
    scatterPlot.x(this.xAccessor, xScale);
    scatterPlot.y(this.yValueAccessor, yScale);
    scatterPlot.attr('fill', (d: any) => this.colorScale.scale(d.name));
    scatterPlot.attr('opacity', 1);
    scatterPlot.size(ChartHelpers.TOOLTIP_CIRCLE_SIZE * 2);
    scatterPlot.datasets([this.lastPointsDataset]);
    this.scatterPlot = scatterPlot;

    let nanDisplay = new Plottable.Plots.Scatter<number|Date, number>();
    nanDisplay.x(this.xAccessor, xScale);
    nanDisplay.y((x) => x.displayY, yScale);
    nanDisplay.attr('fill', (d: any) => this.colorScale.scale(d.name));
    nanDisplay.attr('opacity', 1);
    nanDisplay.size(ChartHelpers.NAN_SYMBOL_SIZE * 2);
    nanDisplay.datasets([this.nanDataset]);
    nanDisplay.symbol(Plottable.SymbolFactories.triangle);
    this.nanDisplay = nanDisplay;

    return new Plottable.Components.Group(
        [nanDisplay, scatterPlot, smoothLinePlot, group]);
  }

  /** Updates the chart when a dataset changes. Called every time the data of
   * a dataset changes to update the charts.
   */
  private _onDatasetChanged(dataset: Plottable.Dataset) {
    if (this.smoothingEnabled) {
      this.resmoothDataset(dataset);
    }
    this.updateSpecialDatasets();
  }

  public ignoreYOutliers(ignoreYOutliers: boolean) {
    if (ignoreYOutliers !== this._ignoreYOutliers) {
      this._ignoreYOutliers = ignoreYOutliers;
      this.updateSpecialDatasets();
      this.resetYDomain();
    }
  }

  /** Constructs special datasets. Each special dataset contains exceptional
   * values from all of the regular datasets, e.g. last points in series, or
   * NaN values. Those points will have a `name` and `relative` property added
   * (since usually those are context in the surrounding dataset).
   */
  private updateSpecialDatasets() {
    const accessor = this.getAccessor();

    let lastPointsData =
        this.datasets
            .map((d) => {
              let datum = null;
              // filter out NaNs to ensure last point is a clean one
              let nonNanData =
                  d.data().filter((x) => !isNaN(accessor(x, -1, d)));
              if (nonNanData.length > 0) {
                let idx = nonNanData.length - 1;
                datum = nonNanData[idx];
                datum.name = d.metadata().name;
                datum.relative = ChartHelpers.relativeAccessor(datum, -1, d);
              }
              return datum;
            })
            .filter((x) => x != null);
    this.lastPointsDataset.data(lastPointsData);

    // Take a dataset, return an array of NaN data points
    // the NaN points will have a "displayY" property which is the
    // y-value of a nearby point that was not NaN (0 if all points are NaN)
    let datasetToNaNData = (d: Plottable.Dataset) => {
      let displayY = null;
      let data = d.data();
      let i = 0;
      while (i < data.length && displayY == null) {
        if (!isNaN(accessor(data[i], -1, d))) {
          displayY = accessor(data[i], -1, d);
        }
        i++;
      }
      if (displayY == null) {
        displayY = 0;
      }
      let nanData = [];
      for (i = 0; i < data.length; i++) {
        if (!isNaN(accessor(data[i], -1, d))) {
          displayY = accessor(data[i], -1, d);
        } else {
          data[i].name = d.metadata().name;
          data[i].displayY = displayY;
          data[i].relative = ChartHelpers.relativeAccessor(data[i], -1, d);
          nanData.push(data[i]);
        }
      }
      return nanData;
    };
    let nanData = _.flatten(this.datasets.map(datasetToNaNData));
    this.nanDataset.data(nanData);
  }

  public resetDomain() {
    this.resetXDomain();
    this.resetYDomain();
  }

  private resetXDomain() {
    let xDomain;
    if (this._defaultXRange != null) {
      // Use the range specified by the caller.
      xDomain = this._defaultXRange;
    } else {
      // (Copied from DragZoomLayer.unzoom.)
      const xScale = this.xScale as any;
      xScale._domainMin = null;
      xScale._domainMax = null;
      xDomain = xScale._getExtent();
    }
    this.xScale.domain(xDomain);
  }

  private resetYDomain() {
    let yDomain;
    if (this._defaultYRange != null) {
      // Use the range specified by the caller.
      yDomain = this._defaultYRange;
    } else {
      // Generate a reasonable range.
      const accessor = this.getAccessor();
      let datasetToValues: (d: Plottable.Dataset) => number[] = (d) => {
        return d.data().map((x) => accessor(x, -1, d));
      };
      let vals = _.flatten(this.datasets.map(datasetToValues));
      vals = vals.filter((x) => x === x && x !== Infinity && x !== -Infinity);
      yDomain = ChartHelpers.computeDomain(vals, this._ignoreYOutliers);
    }
    this.yScale.domain(yDomain);
  }

  private getAccessor(): Plottable.IAccessor<number> {
    return this.smoothingEnabled ? this.smoothedAccessor : this.yValueAccessor;
  }

  private setupTooltips(
      plot: Plottable.XYPlot<number|Date, number>,
      tooltipColumns: ChartHelpers.TooltipColumn[]):
      Plottable.Components.Group {
    let pi = new Plottable.Interactions.Pointer();
    pi.attachTo(plot);
    // PointsComponent is a Plottable Component that will hold the little
    // circles we draw over the closest data points
    let pointsComponent = new Plottable.Component();
    let group = new Plottable.Components.Group([plot, pointsComponent]);

    let hideTooltips = () => {
      this.tooltip.style('opacity', 0);
      this.scatterPlot.attr('opacity', 1);
      pointsComponent.content().selectAll('.point').remove();
    };

    let enabled = true;
    let disableTooltips = () => {
      enabled = false;
      hideTooltips();
    };
    let enableTooltips = () => {
      enabled = true;
    };

    this.dzl.interactionStart(disableTooltips);
    this.dzl.interactionEnd(enableTooltips);

    pi.onPointerMove((p: Plottable.Point) => {
      if (!enabled) {
        return;
      }
      let target: ChartHelpers.Point = {
        x: p.x,
        y: p.y,
        datum: null,
        dataset: null,
      };


      let bbox: SVGRect = (<any>this.gridlines.content().node()).getBBox();

      // pts is the closets point to the tooltip for each dataset
      let pts = plot.datasets()
                    .map((dataset) => this.findClosestPoint(target, dataset))
                    .filter(x => x != null);
      let intersectsBBox = Plottable.Utils.DOM.intersectsBBox;
      // We draw tooltips for points that are NaN, or are currently visible
      let ptsForTooltips = pts.filter(
          (p) => intersectsBBox(p.x, p.y, bbox) ||
              isNaN(this.yValueAccessor(p.datum, 0, p.dataset)));
      // Only draw little indicator circles for the non-NaN points
      let ptsToCircle = ptsForTooltips.filter(
          (p) => !isNaN(this.yValueAccessor(p.datum, 0, p.dataset)));

      let ptsSelection: any =
          pointsComponent.content().selectAll('.point').data(
              ptsToCircle,
              (p: ChartHelpers.Point) => p.dataset.metadata().name);
      if (pts.length !== 0) {
        ptsSelection.enter().append('circle').classed('point', true);
        ptsSelection.attr('r', ChartHelpers.TOOLTIP_CIRCLE_SIZE)
            .attr('cx', (p) => p.x)
            .attr('cy', (p) => p.y)
            .style('stroke', 'none')
            .attr(
                'fill',
                (p) => this.colorScale.scale(p.dataset.metadata().name));
        ptsSelection.exit().remove();
        this.drawTooltips(ptsForTooltips, target, tooltipColumns);
      } else {
        hideTooltips();
      }
    });

    pi.onPointerExit(hideTooltips);

    return group;
  }

  private drawTooltips(
      points: ChartHelpers.Point[],
      target: ChartHelpers.Point,
      tooltipColumns: ChartHelpers.TooltipColumn[]) {
    // Formatters for value, step, and wall_time
    this.scatterPlot.attr('opacity', 0);
    let valueFormatter = ChartHelpers.multiscaleFormatter(
        ChartHelpers.Y_TOOLTIP_FORMATTER_PRECISION);

    let dist = (p: ChartHelpers.Point) =>
        Math.pow(p.x - target.x, 2) + Math.pow(p.y - target.y, 2);
    let closestDist = _.min(points.map(dist));

    let valueSortMethod = this.yValueAccessor;
    if (this.smoothingEnabled) {
      valueSortMethod = this.smoothedAccessor;
    }

    if (this.tooltipSortingMethod === 'ascending') {
      points = _.sortBy(points, (d) => valueSortMethod(d.datum, -1, d.dataset));
    } else if (this.tooltipSortingMethod === 'descending') {
      points = _.sortBy(points, (d) => valueSortMethod(d.datum, -1, d.dataset))
                   .reverse();
    } else if (this.tooltipSortingMethod === 'nearest') {
      points = _.sortBy(points, dist);
    } else {
      // The 'default' sorting method maintains the order of names passed to
      // setVisibleSeries(). However we reverse that order when defining the
      // datasets. So we must call reverse again to restore the order.
      points = points.slice(0).reverse();
    }

    let rows = this.tooltip.select('tbody')
                   .html('')
                   .selectAll('tr')
                   .data(points)
                   .enter()
                   .append('tr');
    // Grey out the point if any of the following are true:
    // - The cursor is outside of the x-extent of the dataset
    // - The point's y value is NaN
    rows.classed('distant', (d) => {
      let firstPoint = d.dataset.data()[0];
      let lastPoint = _.last(d.dataset.data());
      let firstX = this.xScale.scale(this.xAccessor(firstPoint, 0, d.dataset));
      let lastX = this.xScale.scale(this.xAccessor(lastPoint, 0, d.dataset));
      let s = this.smoothingEnabled ?
          d.datum.smoothed : this.yValueAccessor(d.datum, 0, d.dataset);
      return target.x < firstX || target.x > lastX || isNaN(s);
    });
    rows.classed('closest', (p) => dist(p) === closestDist);
    // It is a bit hacky that we are manually applying the width to the swatch
    // and the nowrap property to the text here. The reason is as follows:
    // the style gets updated asynchronously by Polymer scopeSubtree observer.
    // Which means we would get incorrect sizing information since the text
    // would wrap by default. However, we need correct measurements so that
    // we can stop the text from falling off the edge of the screen.
    // therefore, we apply the size-critical styles directly.
    rows.style('white-space', 'nowrap');
    rows.append('td')
        .append('span')
        .classed('swatch', true)
        .style(
            'background-color',
            (d) => this.colorScale.scale(d.dataset.metadata().name));

    _.each(tooltipColumns, (column) => {
      rows.append('td').text(
          (d) => column.evaluate(d, {
            smoothingEnabled: this.smoothingEnabled,
          }));
    });

    // compute left position
    let documentWidth = document.body.clientWidth;
    let node: any = this.tooltip.node();
    let parentRect = node.parentElement.getBoundingClientRect();
    let nodeRect = node.getBoundingClientRect();
    // prevent it from falling off the right side of the screen
    let left = documentWidth - parentRect.left - nodeRect.width - 60, top = 0;

    if (this.tooltipPosition === 'right') {
      left = Math.min(parentRect.width, left);
    } else {  // 'bottom'
      left = Math.min(0, left);
      top = parentRect.height + ChartHelpers.TOOLTIP_Y_PIXEL_OFFSET;
    }

    this.tooltip.style('transform', 'translate(' + left + 'px,' + top + 'px)');
    this.tooltip.style('opacity', 1);
  }

  private findClosestPoint(
      target: ChartHelpers.Point,
      dataset: Plottable.Dataset): ChartHelpers.Point {
    let points: ChartHelpers.Point[] = dataset.data().map((d, i) => {
      let x = this.xAccessor(d, i, dataset);
      let y = this.smoothingEnabled ? this.smoothedAccessor(d, i, dataset) :
                                      this.yValueAccessor(d, i, dataset);
      return {
        x: this.xScale.scale(x),
        y: this.yScale.scale(y),
        datum: d,
        dataset: dataset,
      };
    });
    let idx: number =
        _.sortedIndex(points, target, (p: ChartHelpers.Point) => p.x);
    if (idx === points.length) {
      return points[points.length - 1];
    } else if (idx === 0) {
      return points[0];
    } else {
      let prev = points[idx - 1];
      let next = points[idx];
      let prevDist = Math.abs(prev.x - target.x);
      let nextDist = Math.abs(next.x - target.x);
      return prevDist < nextDist ? prev : next;
    }
  }

  private resmoothDataset(dataset: Plottable.Dataset) {
    let data = dataset.data();
    const smoothingWeight = this.smoothingWeight;
    let last = data.length > 0 ? this.yValueAccessor(data[0], 0, dataset) : NaN;
    data.forEach((d, i) => {
      if (!_.isFinite(last)) {
        d.smoothed = this.yValueAccessor(d, i, dataset);
      } else {
        // 1st-order IIR low-pass filter to attenuate the higher-
        // frequency components of the time-series.
        d.smoothed = last * smoothingWeight + (
            1 - smoothingWeight) * this.yValueAccessor(d, i, dataset);
      }
      last = d.smoothed;
    });
  }

  private getDataset(name: string) {
    if (this.name2datasets[name] === undefined) {
      this.name2datasets[name] = new Plottable.Dataset([], {name: name});
    }
    return this.name2datasets[name];
  }

  static getYScaleFromType(yScaleType: string):
      Plottable.QuantitativeScale<number> {
    if (yScaleType === 'log') {
      return new Plottable.Scales.ModifiedLog();
    } else if (yScaleType === 'linear') {
      return new Plottable.Scales.Linear();
    } else {
      throw new Error('Unrecognized yScale type ' + yScaleType);
    }
  }

  /**
   * Update the selected series on the chart.
   */
  public setVisibleSeries(names: string[]) {
    names = names.sort();
    this.seriesNames = names;

    names.reverse();  // draw first series on top
    this.datasets.forEach((d) => d.offUpdate(this.onDatasetChanged));
    this.datasets = names.map((r) => this.getDataset(r));
    this.datasets.forEach((d) => d.onUpdate(this.onDatasetChanged));
    this.linePlot.datasets(this.datasets);

    if (this.smoothingEnabled) {
      this.smoothLinePlot.datasets(this.datasets);
    }
    this.updateSpecialDatasets();
  }

  /**
   * Set the data of a series on the chart.
   */
  public setSeriesData(name: string, data: ChartHelpers.ScalarDatum[]) {
    this.getDataset(name).data(data);
  }

  public smoothingUpdate(weight: number) {
    this.smoothingWeight = weight;
    this.datasets.forEach((d) => this.resmoothDataset(d));

    if (!this.smoothingEnabled) {
      this.linePlot.addClass('ghost');
      this.scatterPlot.y(this.smoothedAccessor, this.yScale);
      this.smoothingEnabled = true;
      this.smoothLinePlot.datasets(this.datasets);
    }

    this.updateSpecialDatasets();
  }

  public smoothingDisable() {
    if (this.smoothingEnabled) {
      this.linePlot.removeClass('ghost');
      this.scatterPlot.y(this.yValueAccessor, this.yScale);
      this.smoothLinePlot.datasets([]);
      this.smoothingEnabled = false;
      this.updateSpecialDatasets();
    }
  }

  public setTooltipSortingMethod(method: string) {
    this.tooltipSortingMethod = method;
  }

  public setTooltipPosition(position: string) {
    this.tooltipPosition = position;
  }

  public renderTo(targetSVG: d3.Selection<any, any, any, any>) {
    this.targetSVG = targetSVG;
    this.outer.renderTo(targetSVG);

    if (this._defaultXRange != null) {
      // A higher-level component provided a default range for the X axis.
      // Start with that range.
      this.resetXDomain();
    }

    if (this._defaultYRange != null) {
      // A higher-level component provided a default range for the Y axis.
      // Start with that range.
      this.resetYDomain();
    }
  }

  public redraw() {
    this.outer.redraw();
  }

  public destroy() {
    this.outer.destroy();
  }
}
