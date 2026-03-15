'use client';

import { useRef, useEffect, useMemo, useCallback } from 'react';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';
import type { GraphData, GraphLink, GraphNode } from '../lib/api';

const RISK_NODE_COLORS = {
  normal: '#3772ff',
  suspicious: '#f97316',
  laundering: '#ef4444',
} as const;

const labelSpriteCache = new Map<string, THREE.SpriteMaterial>();

type SimulationNode = GraphNode & {
  x?: number;
  y?: number;
  z?: number;
  vx?: number;
  vy?: number;
  vz?: number;
};

type SimulationLink = GraphLink & {
  source: string | SimulationNode;
  target: string | SimulationNode;
};

type ForceGraphHandle = {
  d3Force: (name: string, force?: ((alpha: number) => void)) => {
    strength?: (value: number) => void;
    distance?: (value: number) => void;
  } | undefined;
  d3ReheatSimulation: () => void;
  cameraPosition: (
    position: { x: number; y: number; z: number },
    lookAt?: SimulationNode,
    ms?: number
  ) => void;
};

type NetworkGraphProps = {
  data: GraphData;
  selectedNode: GraphNode | null;
  onNodeClick: (node: GraphNode) => void;
  isDarkMode: boolean;
};

function getLabelMaterial(text: string, isDarkMode: boolean) {
  const cacheKey = `${text}:${isDarkMode ? 'dark' : 'light'}`;
  const cached = labelSpriteCache.get(cacheKey);
  if (cached) return cached;

  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 160;
  const ctx = canvas.getContext('2d');

  if (ctx) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = 'bold 88px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 12;
    ctx.strokeStyle = isDarkMode ? 'rgba(2, 6, 23, 0.9)' : 'rgba(255, 255, 255, 0.95)';
    ctx.fillStyle = isDarkMode ? '#f8fafc' : '#0f172a';
    ctx.strokeText(text, canvas.width / 2, canvas.height / 2);
    ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.needsUpdate = true;
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true, depthWrite: false });
  labelSpriteCache.set(cacheKey, material);
  return material;
}

function computeClusters(nodes: SimulationNode[], links: SimulationLink[]) {
  const adj = new Map<string, Set<string>>();
  for (const n of nodes) adj.set(n.id, new Set());
  for (const l of links) {
    const s = typeof l.source === 'object' ? l.source.id : l.source;
    const t = typeof l.target === 'object' ? l.target.id : l.target;
    adj.get(s)?.add(t);
    adj.get(t)?.add(s);
  }

  const clusterMap = new Map<string, number>();
  let clusterId = 0;
  const visited = new Set<string>();
  const sortedNodes = [...nodes].sort((a, b) => (adj.get(b.id)?.size ?? 0) - (adj.get(a.id)?.size ?? 0));

  for (const node of sortedNodes) {
    if (visited.has(node.id)) continue;
    const queue = [node.id];
    visited.add(node.id);
    while (queue.length > 0) {
      const current = queue.shift()!;
      clusterMap.set(current, clusterId);
      for (const neighbor of adj.get(current) ?? []) {
        if (!visited.has(neighbor)) {
          visited.add(neighbor);
          queue.push(neighbor);
        }
      }
    }
    clusterId++;
  }

  return clusterMap;
}

function computeDegrees(nodes: SimulationNode[], links: SimulationLink[]) {
  const degreeMap = new Map<string, number>();
  for (const n of nodes) degreeMap.set(n.id, 0);
  for (const l of links) {
    const s = typeof l.source === 'object' ? l.source.id : l.source;
    const t = typeof l.target === 'object' ? l.target.id : l.target;
    degreeMap.set(s, (degreeMap.get(s) ?? 0) + 1);
    degreeMap.set(t, (degreeMap.get(t) ?? 0) + 1);
  }
  return degreeMap;
}

export default function NetworkGraph({ data, selectedNode, onNodeClick, isDarkMode }: NetworkGraphProps) {
  const fgRef = useRef<ForceGraphHandle | null>(null);

  const clusterMap = useMemo(() => computeClusters(data.nodes, data.links), [data]);
  const degreeMap = useMemo(() => computeDegrees(data.nodes, data.links), [data]);
  const maxDegree = useMemo(() => Math.max(1, ...degreeMap.values()), [degreeMap]);

  const topLaunderingNodeIds = useMemo(() => {
    return new Set(
      data.nodes
        .filter((node) => node.risk === 'laundering')
        .sort((a, b) => {
          const riskDelta = (b.riskScore ?? -Infinity) - (a.riskScore ?? -Infinity);
          if (riskDelta !== 0) return riskDelta;
          const txDelta = (b.txCount ?? 0) - (a.txCount ?? 0);
          if (txDelta !== 0) return txDelta;
          return (degreeMap.get(b.id) ?? 0) - (degreeMap.get(a.id) ?? 0);
        })
        .slice(0, 3)
        .map((node) => node.id)
    );
  }, [data.nodes, degreeMap]);

  const { highlightNodes, highlightLinks } = useMemo(() => {
    const newNodes = new Set<string>();
    const newLinks = new Set<SimulationLink>();

    if (selectedNode) {
      newNodes.add(selectedNode.id);
      data.links.forEach((link) => {
        const s = typeof link.source === 'object' ? link.source.id : link.source;
        const t = typeof link.target === 'object' ? link.target.id : link.target;
        if (s === selectedNode.id || t === selectedNode.id) {
          newLinks.add(link);
          newNodes.add(s);
          newNodes.add(t);
        }
      });
    }

    return { highlightNodes: newNodes, highlightLinks: newLinks };
  }, [selectedNode, data]);

  useEffect(() => {
    const fg = fgRef.current;
    if (!fg) return;

    fg.d3Force('cluster', (alpha: number) => {
      const centroids = new Map<number, { x: number; y: number; z: number; count: number }>();

      for (const node of data.nodes) {
        const c = clusterMap.get(node.id) ?? 0;
        if (!centroids.has(c)) centroids.set(c, { x: 0, y: 0, z: 0, count: 0 });
        const centroid = centroids.get(c)!;
        centroid.x += node.x ?? 0;
        centroid.y += node.y ?? 0;
        centroid.z += node.z ?? 0;
        centroid.count++;
      }

      for (const [, value] of centroids) {
        value.x /= value.count;
        value.y /= value.count;
        value.z /= value.count;
      }

      const strength = 0.3 * alpha;
      for (const node of data.nodes) {
        const c = clusterMap.get(node.id) ?? 0;
        const centroid = centroids.get(c);
        if (!centroid) continue;
        node.vx = (node.vx ?? 0) + (centroid.x - (node.x ?? 0)) * strength;
        node.vy = (node.vy ?? 0) + (centroid.y - (node.y ?? 0)) * strength;
        node.vz = (node.vz ?? 0) + (centroid.z - (node.z ?? 0)) * strength;
      }
    });

    fg.d3Force('charge')?.strength(-160);
    fg.d3Force('link')?.distance(45);
    fg.d3ReheatSimulation();
  }, [data, clusterMap]);

  const getNodeColor = useCallback((node: GraphNode) => {
    return RISK_NODE_COLORS[node.risk as keyof typeof RISK_NODE_COLORS] ?? RISK_NODE_COLORS.normal;
  }, []);

  const getNodeSize = useCallback((node: GraphNode) => {
    const degree = degreeMap.get(node.id) ?? 1;
    return 2.5 + (degree / maxDegree) * 12;
  }, [degreeMap, maxDegree]);

  const handleNodeClick = useCallback((node: SimulationNode) => {
    if (!fgRef.current) return;

    if (selectedNode?.id !== node.id) {
      const distance = 120;
      const distRatio = 1 + distance / Math.max(1, Math.hypot(node.x ?? 0, node.y ?? 0, node.z ?? 0));

      fgRef.current.cameraPosition(
        {
          x: (node.x ?? 0) * distRatio,
          y: (node.y ?? 0) * distRatio,
          z: (node.z ?? 0) * distRatio,
        },
        node,
        1000
      );
    }

    onNodeClick(node);
  }, [selectedNode, onNodeClick]);

  const renderNode = useCallback((node: SimulationNode) => {
    const isSelected = selectedNode?.id === node.id;
    const isHighlighted = highlightNodes.has(node.id);
    const isDimmed = Boolean(selectedNode) && !isHighlighted;
    const size = getNodeSize(node);
    const color = getNodeColor(node);

    const group = new THREE.Group();

    const sphere = new THREE.Mesh(
      new THREE.SphereGeometry(size, 18, 18),
      new THREE.MeshStandardMaterial({
        color: isDimmed ? (isDarkMode ? '#1e293b' : '#94a3b8') : color,
        transparent: true,
        opacity: isDimmed ? 0.2 : 1,
        emissive: isDimmed ? '#000000' : color,
        emissiveIntensity: isSelected ? 0.75 : isHighlighted ? 0.35 : 0.18,
        roughness: 0.35,
        metalness: 0.1,
      })
    );
    group.add(sphere);

    if ((isSelected || isHighlighted) && !isDimmed) {
      const aura = new THREE.Mesh(
        new THREE.SphereGeometry(size * 1.8, 16, 16),
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity: isSelected ? 0.16 : 0.08,
        })
      );
      group.add(aura);
    }

    if (topLaunderingNodeIds.has(node.id) && !isDimmed) {
      const label = new THREE.Sprite(getLabelMaterial(node.id, isDarkMode));
      label.position.set(0, size + 8, 0);
      label.scale.set(32, 10, 1);
      group.add(label);
    }

    return group;
  }, [selectedNode, highlightNodes, getNodeSize, getNodeColor, isDarkMode, topLaunderingNodeIds]);

  return (
    <div className="w-full h-full transition-colors duration-300 cursor-move">
      <ForceGraph3D
        ref={fgRef}
        graphData={data}
        backgroundColor={isDarkMode ? '#020617' : '#FFFEEF'}
        showNavInfo={false}
        nodeThreeObject={renderNode}
        nodeLabel={(node: SimulationNode) => topLaunderingNodeIds.has(node.id) ? node.id : ''}
        linkColor={(link: SimulationLink) => {
          if (highlightLinks.has(link)) return isDarkMode ? '#F7F3C5' : '#322035';
          return isDarkMode ? '#F7F3C5' : '#322035';
        }}
        linkOpacity={0.45}
        linkWidth={(link: SimulationLink) => highlightLinks.has(link) ? 2 : 0.4}
        linkDirectionalParticles={(link: SimulationLink) => highlightLinks.has(link) ? 3 : 0}
        linkDirectionalParticleWidth={2}
        linkDirectionalParticleSpeed={0.005}
        onNodeClick={handleNodeClick}
        enableNodeDrag={false}
        cooldownTicks={200}
        warmupTicks={100}
      />
    </div>
  );
}
