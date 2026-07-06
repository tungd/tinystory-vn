import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Legend,
  ResponsiveContainer,
} from 'recharts';

export interface EvalScores {
  grammar: number;
  creativity: number;
  moral_clarity: number;
  prompt_adherence: number;
}

export interface EvalSeries {
  name: string;
  scores: EvalScores;
}

export interface EvalRadarProps {
  series: EvalSeries[];
}

const AXIS_LABELS: Record<keyof EvalScores, string> = {
  grammar: 'Grammar',
  creativity: 'Creativity',
  moral_clarity: 'Moral Clarity',
  prompt_adherence: 'Prompt Adherence',
};

const COLORS = ['#2563eb', '#16a34a'];

export function EvalRadar({ series }: EvalRadarProps) {
  const axes = Object.keys(AXIS_LABELS) as (keyof EvalScores)[];

  const chartData = axes.map((axis) => {
    const point: Record<string, string | number> = { axis: AXIS_LABELS[axis] };
    series.forEach((s) => {
      point[s.name] = s.scores[axis];
    });
    return point;
  });

  return (
    <ResponsiveContainer width="100%" height={260}>
      <RadarChart data={chartData} aria-label="Evaluation radar chart">
        <PolarGrid />
        <PolarAngleAxis dataKey="axis" tick={{ fontSize: 12 }} />
        <PolarRadiusAxis domain={[0, 10]} tick={{ fontSize: 10 }} tickCount={6} />
        {series.map((s, i) => (
          <Radar
            key={s.name}
            name={s.name}
            dataKey={s.name}
            stroke={COLORS[i % COLORS.length]}
            fill={COLORS[i % COLORS.length]}
            fillOpacity={0.15}
          />
        ))}
        {series.length > 1 && <Legend />}
      </RadarChart>
    </ResponsiveContainer>
  );
}
