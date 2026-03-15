export interface GraphNode {
  id: string;
  risk: 'normal' | 'suspicious' | 'laundering';
  txCount: number;
  pattern: string;
  aiExplanation: string;
  role?: string;
  riskScore?: number;
  cluster?: number;
}

export interface GraphLink {
  source: string;
  target: string;
  amount: number;
}

export interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

/**
 * Parse a CSV string handling quoted fields (e.g. aiExplanation with commas).
 * Returns an array of objects keyed by header names.
 */
function parseCSV(text: string): Record<string, string>[] {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return [];

  const headers = parseCsvLine(lines[0]);
  const rows: Record<string, string>[] = [];

  for (let i = 1; i < lines.length; i++) {
    const values = parseCsvLine(lines[i]);
    const row: Record<string, string> = {};
    for (let j = 0; j < headers.length; j++) {
      row[headers[j]] = values[j] ?? '';
    }
    rows.push(row);
  }
  return rows;
}

/** Parse a single CSV line respecting quoted fields. */
function parseCsvLine(line: string): string[] {
  const fields: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (inQuotes) {
      if (ch === '"') {
        if (i + 1 < line.length && line[i + 1] === '"') {
          current += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        current += ch;
      }
    } else {
      if (ch === '"') {
        inQuotes = true;
      } else if (ch === ',') {
        fields.push(current);
        current = '';
      } else {
        current += ch;
      }
    }
  }
  fields.push(current);
  return fields;
}

function toRisk(value: string): 'normal' | 'suspicious' | 'laundering' {
  if (value === 'laundering' || value === 'suspicious') return value;
  return 'normal';
}

export async function loadGraphFromCSV(): Promise<GraphData> {
  const [nodesRes, edgesRes] = await Promise.all([
    fetch('/node_data/nodes.csv'),
    fetch('/node_data/edges.csv'),
  ]);

  if (!nodesRes.ok) throw new Error('Could not load nodes.csv');
  if (!edgesRes.ok) throw new Error('Could not load edges.csv');

  const nodesText = await nodesRes.text();
  const edgesText = await edgesRes.text();

  const nodesRaw = parseCSV(nodesText);
  const edgesRaw = parseCSV(edgesText);

  const nodes: GraphNode[] = nodesRaw.map((r) => ({
    id: r.id,
    risk: toRisk(r.risk),
    txCount: parseInt(r.txCount, 10) || 0,
    pattern: r.pattern || 'None',
    aiExplanation: r.aiExplanation || 'No anomalies detected.',
    role: r.role || undefined,
    riskScore: r.riskScore ? parseFloat(r.riskScore) : undefined,
    cluster: r.cluster ? parseInt(r.cluster, 10) : undefined,
  }));

  const links: GraphLink[] = edgesRaw.map((r) => ({
    source: r.source,
    target: r.target,
    amount: parseFloat(r.amount) || 0,
  }));

  return { nodes, links };
}
