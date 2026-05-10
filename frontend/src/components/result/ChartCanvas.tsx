'use client';

import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { cn } from '@/lib/utils';

// ─── Types ───────────────────────────────────────────────────────────────────

export type ChartType = 'line' | 'bar' | 'table' | 'stat';

export interface ChartCanvasProps {
  chartType: ChartType;
  /** Records-orientation data — array of objects keyed by column name */
  data: Array<Record<string, string | number | boolean | null>>;
  /** Column names in the order they should appear in tables */
  columns: string[];
  /** Key for x-axis (line, bar). Optional — falls back to first column */
  xKey?: string;
  /** Key for y-axis (line, bar) and the stat value (stat). Optional — falls back to first numeric column */
  yKey?: string;
  /**
   * Optional metric name driving this figure. When present, becomes the FIG label.
   * When absent, the FIG label falls back to the y-axis column name, then the chart type.
   */
  metricName?: string;
  /** Extra classes for the outer wrapper */
  className?: string;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function firstNumericKey(
  data: Array<Record<string, string | number | boolean | null>>,
  columns: string[],
): string | undefined {
  if (!data.length) return undefined;
  return columns.find((col) => typeof data[0][col] === 'number');
}

function formatNumber(value: number): string {
  return new Intl.NumberFormat('en-GB', { maximumFractionDigits: 2 }).format(value);
}

function formatValue(value: unknown): string {
  if (typeof value === 'number') return formatNumber(value);
  if (value === null || value === undefined) return '—';
  return String(value);
}

function isMasked(value: unknown): boolean {
  if (typeof value !== 'string') return false;
  // Detect both legacy block-char masking and the bullet pattern.
  return value.includes('█') || /•{2,}/.test(value);
}

function isNumericColumn(
  col: string,
  data: Array<Record<string, string | number | boolean | null>>,
): boolean {
  return data.some((row) => typeof row[col] === 'number');
}

function prettyColumn(col: string): string {
  return col.replace(/_/g, ' ');
}

/**
 * Compute clean round-number Y-axis ticks for a given data max.
 * 3.8M → [0, 1M, 2M, 3M, 4M] not [0, 950K, 1.9M, 2.9M, 3.8M].
 * 1240 → [0, 400, 800, 1200] not [0, 310, 620, 930, 1240].
 * The d3-style "nice" algorithm — separates a studied chart from a default one.
 */
function calculateNiceTicks(maxValue: number, targetCount = 6): number[] {
  if (maxValue <= 0) return [0];
  const rawStep = maxValue / (targetCount - 1);
  const magnitude = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const normalized = rawStep / magnitude;
  let niceStep: number;
  if (normalized <= 1) niceStep = 1 * magnitude;
  else if (normalized <= 2) niceStep = 2 * magnitude;
  else if (normalized <= 2.5) niceStep = 2.5 * magnitude;
  else if (normalized <= 5) niceStep = 5 * magnitude;
  else niceStep = 10 * magnitude;

  const ticks: number[] = [];
  let v = 0;
  while (v <= maxValue + niceStep * 0.001) {
    ticks.push(v);
    v += niceStep;
  }
  return ticks;
}

function dataMax(
  data: Array<Record<string, string | number | boolean | null>>,
  yKey: string,
): number {
  return Math.max(
    0,
    ...data.map((d) => (typeof d[yKey] === 'number' ? (d[yKey] as number) : 0)),
  );
}

// ─── Custom Tooltip ───────────────────────────────────────────────────────────

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ value: number | string; dataKey: string }>;
  label?: string;
  yKey: string;
}

function CustomTooltip({ active, payload, label, yKey }: CustomTooltipProps) {
  if (!active || !payload?.length) return null;
  const val = payload[0]?.value;
  return (
    <div className="bg-popover text-popover-foreground border border-border rounded-md px-3 py-2 text-xs">
      <p className="font-medium mb-0.5">{label}</p>
      <p className="tabular-nums">
        <span className="text-muted-foreground">{prettyColumn(yKey)} </span>
        <span>{typeof val === 'number' ? formatNumber(val) : val}</span>
      </p>
    </div>
  );
}

// ─── ChartLine ────────────────────────────────────────────────────────────────

interface LineBarProps {
  data: Array<Record<string, string | number | boolean | null>>;
  xKey: string;
  yKey: string;
}

function ChartLine({ data, xKey, yKey }: LineBarProps) {
  const ticks = calculateNiceTicks(dataMax(data, yKey));
  const domainMax = ticks[ticks.length - 1];

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
        <CartesianGrid
          stroke="var(--border)"
          strokeDasharray="3 3"
          strokeOpacity={0.3}
          vertical={false}
        />
        <XAxis
          dataKey={xKey}
          axisLine={false}
          tickLine={false}
          tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
          dy={6}
        />
        <YAxis
          axisLine={false}
          tickLine={false}
          tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
          tickFormatter={(v: number) =>
            new Intl.NumberFormat('en-GB', {
              notation: 'compact',
              maximumFractionDigits: 1,
            }).format(v)
          }
          width={44}
          ticks={ticks}
          domain={[0, domainMax]}
        />
        <Tooltip
          content={<CustomTooltip yKey={yKey} />}
          cursor={{ stroke: 'var(--border)', strokeWidth: 1, strokeDasharray: '3 3' }}
        />
        <Line
          type="linear"
          dataKey={yKey}
          stroke="var(--chart-1)"
          strokeWidth={1.75}
          dot={false}
          activeDot={{ r: 4, fill: 'var(--chart-1)', strokeWidth: 0 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ─── ChartBar ─────────────────────────────────────────────────────────────────

function ChartBar({ data, xKey, yKey }: LineBarProps) {
  const ticks = calculateNiceTicks(dataMax(data, yKey));
  const domainMax = ticks[ticks.length - 1];

  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        data={data}
        margin={{ top: 8, right: 8, bottom: 8, left: 8 }}
        barCategoryGap="28%"
      >
        <CartesianGrid
          stroke="var(--border)"
          strokeDasharray="3 3"
          strokeOpacity={0.3}
          vertical={false}
        />
        <XAxis
          dataKey={xKey}
          axisLine={false}
          tickLine={false}
          tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
          dy={6}
        />
        <YAxis
          axisLine={false}
          tickLine={false}
          tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
          tickFormatter={(v: number) =>
            new Intl.NumberFormat('en-GB', {
              notation: 'compact',
              maximumFractionDigits: 1,
            }).format(v)
          }
          width={44}
          ticks={ticks}
          domain={[0, domainMax]}
        />
        <Tooltip
          content={<CustomTooltip yKey={yKey} />}
          cursor={{ fill: 'var(--muted)', fillOpacity: 0.4 }}
        />
        <Bar dataKey={yKey} fill="var(--chart-1)" radius={[2, 2, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  );
}

// ─── ChartTable ───────────────────────────────────────────────────────────────

interface TableProps {
  data: Array<Record<string, string | number | boolean | null>>;
  columns: string[];
}

const MAX_ROWS = 10;

function ChartTable({ data, columns }: TableProps) {
  const visibleRows = data.slice(0, MAX_ROWS);
  const hasMore = data.length > MAX_ROWS;

  return (
    <div className="w-full">
      <table className="w-full">
        <thead>
          <tr className="border-b border-border">
            {columns.map((col) => (
              <th
                key={col}
                className={cn(
                  'text-[11px] uppercase tracking-wide font-medium text-muted-foreground',
                  'px-3 py-2.5 whitespace-nowrap',
                  isNumericColumn(col, data) ? 'text-right' : 'text-left',
                )}
              >
                {prettyColumn(col)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {visibleRows.map((row, i) => (
            <tr
              key={i}
              className={cn(
                'border-b border-border/40',
                i === visibleRows.length - 1 && 'border-b-0',
              )}
            >
              {columns.map((col) => {
                const numeric = isNumericColumn(col, data);
                const masked = isMasked(row[col]);
                return (
                  <td
                    key={col}
                    className={cn(
                      'text-sm py-3 px-3',
                      numeric ? 'text-right tabular-nums' : 'text-left',
                      masked
                        ? 'text-muted-foreground font-mono tracking-widest whitespace-nowrap'
                        : 'text-foreground',
                    )}
                  >
                    {formatValue(row[col])}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {hasMore && (
        <p className="text-xs text-muted-foreground mt-3 px-3">
          Showing {MAX_ROWS} of {data.length} rows
        </p>
      )}
    </div>
  );
}

// ─── ChartStat ────────────────────────────────────────────────────────────────

interface StatProps {
  data: Array<Record<string, string | number | boolean | null>>;
  columns: string[];
  yKey?: string;
}

function ChartStat({ data, columns, yKey }: StatProps) {
  const key = yKey ?? firstNumericKey(data, columns) ?? columns[0];
  const rawValue = data[0]?.[key] ?? null;

  const formatted =
    typeof rawValue === 'number'
      ? new Intl.NumberFormat('en-GB').format(rawValue)
      : rawValue !== null && rawValue !== undefined
      ? String(rawValue)
      : '—';

  // The FIG label at top of the card already names the column.
  // The bottom label here would be redundant in every case — drop it.
  return (
    <div className="flex items-baseline">
      <span className="font-sans font-medium text-7xl text-foreground tabular-nums leading-none tracking-tighter">
        {formatted}
      </span>
    </div>
  );
}

// ─── FigLabel ─────────────────────────────────────────────────────────────────

/**
 * Editorial top-left anchor for every chart card — the single move that
 * lifts the figure from "rendered chart" to "studied artifact". Mono
 * uppercase, low contrast, sits inside the card-surface padding.
 */
function FigLabel({ label }: { label: string }) {
  return (
    <div className="text-[11px] font-mono uppercase tracking-[0.08em] text-foreground/40 mb-3">
      FIG · {label}
    </div>
  );
}

// ─── ChartCanvas (main export) ────────────────────────────────────────────────

export function ChartCanvas({
  chartType,
  data,
  columns,
  xKey,
  yKey,
  metricName,
  className,
}: ChartCanvasProps) {
  const resolvedXKey = xKey ?? columns[0];
  const resolvedYKey = yKey ?? firstNumericKey(data, columns) ?? columns[1] ?? columns[0];

  const figLabel = (metricName ?? resolvedYKey ?? chartType).toUpperCase();

  const isChart = chartType === 'line' || chartType === 'bar';

  return (
    <div className={cn(className)}>
      {isChart && (
        <div className="card-surface w-full p-5">
          <FigLabel label={figLabel} />
          <div className="w-full h-72">
            {chartType === 'line' && (
              <ChartLine data={data} xKey={resolvedXKey} yKey={resolvedYKey} />
            )}
            {chartType === 'bar' && (
              <ChartBar data={data} xKey={resolvedXKey} yKey={resolvedYKey} />
            )}
          </div>
        </div>
      )}
      {chartType === 'table' && (
        <div className="card-surface w-full p-5">
          <FigLabel label={figLabel} />
          <ChartTable data={data} columns={columns} />
        </div>
      )}
      {chartType === 'stat' && (
        <div className="card-surface w-full p-8">
          <FigLabel label={figLabel} />
          <ChartStat data={data} columns={columns} yKey={yKey} />
        </div>
      )}
    </div>
  );
}

// ─── Preview (default export for v0) ─────────────────────────────────────────

export default function ChartCanvasPreview() {
  const lineData = [
    { month: 'Jan', revenue: 120000 },
    { month: 'Feb', revenue: 145000 },
    { month: 'Mar', revenue: 138000 },
    { month: 'Apr', revenue: 162000 },
    { month: 'May', revenue: 178000 },
    { month: 'Jun', revenue: 195000 },
  ];
  const barData = [
    { region: 'London', customers: 1240 },
    { region: 'Manchester', customers: 870 },
    { region: 'Edinburgh', customers: 560 },
    { region: 'Bristol', customers: 420 },
    { region: 'Cardiff', customers: 310 },
  ];
  const tableData = [
    { customer_id: 'C001', name: 'Acme Ltd', balance: 482000, risk: 'LOW' },
    { customer_id: 'C002', name: 'Beta Corp', balance: 391000, risk: 'MEDIUM' },
    { customer_id: 'C003', name: 'Gamma Inc', balance: 287000, risk: 'LOW' },
    { customer_id: 'C004', name: 'Delta plc', balance: 198000, risk: 'HIGH' },
    { customer_id: 'C005', name: 'Epsilon Ltd', balance: 156000, risk: 'MEDIUM' },
  ];
  const statData = [{ total_balance: 4200000 }];

  return (
    <div className="min-h-screen bg-background py-12 flex flex-col gap-12">
      <div className="px-6">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground mb-4">Line</p>
        <ChartCanvas
          chartType="line"
          data={lineData}
          columns={['month', 'revenue']}
          xKey="month"
          yKey="revenue"
        />
      </div>
      <div className="px-6">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground mb-4">Bar</p>
        <ChartCanvas
          chartType="bar"
          data={barData}
          columns={['region', 'customers']}
          xKey="region"
          yKey="customers"
        />
      </div>
      <div className="px-6">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground mb-4">Table</p>
        <ChartCanvas
          chartType="table"
          data={tableData}
          columns={['customer_id', 'name', 'balance', 'risk']}
        />
      </div>
      <div className="px-6">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground mb-4">Stat</p>
        <ChartCanvas
          chartType="stat"
          data={statData}
          columns={['total_balance']}
          yKey="total_balance"
        />
      </div>
    </div>
  );
}