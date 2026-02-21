/**
 * RiskTrend.jsx
 * Longitudinal Risk Trajectory Chart Component
 * Displays how patient's risk changes across different trimesters
 * Paper Section: III-D. Dynamic Profiler
 */

export default function RiskTrend({ timelinePoints = [], currentRisk = null, gestationalWeeks = null }) {
  const width = 100;
  const height = 200;
  const padding = { top: 20, right: 10, bottom: 30, left: 35 };

  // Combine timeline points with current risk if available
  const allPoints = [...timelinePoints];
  if (currentRisk && gestationalWeeks !== null) {
    allPoints.push({
      gestational_weeks: gestationalWeeks,
      risk_percent: currentRisk,
      timestamp: new Date().toISOString(),
      isCurrent: true
    });
  }

  if (allPoints.length === 0) {
    return (
      <div className="risk-trend-empty">
        <p className="muted">No timeline data available. Run risk assessment to generate trajectory.</p>
      </div>
    );
  }

  // Sort by gestational weeks
  const sortedPoints = [...allPoints].sort((a, b) => a.gestational_weeks - b.gestational_weeks);

  const xs = sortedPoints.map((p) => p.gestational_weeks);
  const ys = sortedPoints.map((p) => p.risk_percent);
  const minX = Math.max(0, Math.min(...xs) - 2);
  const maxX = Math.min(42, Math.max(...xs) + 2);
  const minY = 0;
  const maxY = 100;

  const scaleX = (x) =>
    padding.left + ((x - minX) / (maxX - minX || 1)) * (width - padding.left - padding.right);
  const scaleY = (y) =>
    height - padding.bottom - ((y - minY) / (maxY - minY || 1)) * (height - padding.top - padding.bottom);

  // Build path for line chart
  const pathData = sortedPoints
    .map((p, idx) => `${idx === 0 ? "M" : "L"} ${scaleX(p.gestational_weeks)} ${scaleY(p.risk_percent)}`)
    .join(" ");

  // Risk zones
  const lowZone = scaleY(40);
  const moderateZone = scaleY(70);

  return (
    <div className="risk-trend-container">
      <h3 className="risk-trend-title">Risk Trajectory</h3>
      <svg className="risk-trend-chart" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        {/* Background risk zones */}
        <rect
          x={padding.left}
          y={moderateZone}
          width={width - padding.left - padding.right}
          height={height - padding.bottom - moderateZone}
          fill="#fee2e2"
          opacity="0.3"
        />
        <rect
          x={padding.left}
          y={lowZone}
          width={width - padding.left - padding.right}
          height={moderateZone - lowZone}
          fill="#fef3c7"
          opacity="0.3"
        />
        <rect
          x={padding.left}
          y={padding.top}
          width={width - padding.left - padding.right}
          height={lowZone - padding.top}
          fill="#d1fae5"
          opacity="0.3"
        />

        {/* Grid lines */}
        {[0, 25, 50, 75, 100].map((y) => (
          <line
            key={`grid-${y}`}
            x1={padding.left}
            y1={scaleY(y)}
            x2={width - padding.right}
            y2={scaleY(y)}
            stroke="#e5dcd4"
            strokeWidth="0.5"
            strokeDasharray="2,2"
          />
        ))}

        {/* Axes */}
        <line
          x1={padding.left}
          y1={padding.top}
          x2={padding.left}
          y2={height - padding.bottom}
          stroke="#ccbdb1"
          strokeWidth="1"
        />
        <line
          x1={padding.left}
          y1={height - padding.bottom}
          x2={width - padding.right}
          y2={height - padding.bottom}
          stroke="#ccbdb1"
          strokeWidth="1"
        />

        {/* Risk trajectory line */}
        {sortedPoints.length > 1 && (
          <path
            d={pathData}
            fill="none"
            stroke="#0f4c5c"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}

        {/* Data points */}
        {sortedPoints.map((p, idx) => (
          <g key={`point-${idx}`}>
            <circle
              cx={scaleX(p.gestational_weeks)}
              cy={scaleY(p.risk_percent)}
              r={p.isCurrent ? "3.5" : "2.5"}
              fill={p.isCurrent ? "#e85d04" : "#0f4c5c"}
              stroke="#fff"
              strokeWidth="1"
            />
            {p.isCurrent && (
              <text
                x={scaleX(p.gestational_weeks)}
                y={scaleY(p.risk_percent) - 8}
                fontSize="8"
                fill="#e85d04"
                fontWeight="600"
                textAnchor="middle"
              >
                Current
              </text>
            )}
          </g>
        ))}

        {/* Y-axis labels */}
        {[0, 25, 50, 75, 100].map((y) => (
          <text
            key={`y-label-${y}`}
            x={padding.left - 5}
            y={scaleY(y) + 3}
            fontSize="9"
            fill="#6d5f55"
            textAnchor="end"
          >
            {y}%
          </text>
        ))}

        {/* X-axis labels */}
        {[0, 12, 24, 36].map((x) => {
          if (x < minX || x > maxX) return null;
          return (
            <text
              key={`x-label-${x}`}
              x={scaleX(x)}
              y={height - padding.bottom + 15}
              fontSize="9"
              fill="#6d5f55"
              textAnchor="middle"
            >
              {x}w
            </text>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="risk-trend-legend">
        <div className="legend-item">
          <span className="legend-dot low" />
          <span>Low (0-40%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot moderate" />
          <span>Moderate (41-70%)</span>
        </div>
        <div className="legend-item">
          <span className="legend-dot high" />
          <span>High (71-100%)</span>
        </div>
      </div>
    </div>
  );
}
