import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

/**
 * 3D rotating neural network visualization using Three.js / React Three Fiber.
 */

function NeuralNode({ position, color, size = 0.08 }) {
  const meshRef = useRef();

  useFrame(({ clock }) => {
    if (meshRef.current) {
      meshRef.current.material.opacity = 0.5 + Math.sin(clock.elapsedTime * 2 + position[0]) * 0.3;
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[size, 16, 16]} />
      <meshStandardMaterial
        color={color}
        transparent
        opacity={0.7}
        emissive={color}
        emissiveIntensity={0.5}
      />
    </mesh>
  );
}

function NeuralConnection({ start, end, color = '#00d4ff', opacity = 0.15 }) {
  const points = useMemo(() => [
    new THREE.Vector3(...start),
    new THREE.Vector3(...end),
  ], [start, end]);

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry().setFromPoints(points);
    return geo;
  }, [points]);

  return (
    <line geometry={geometry}>
      <lineBasicMaterial color={color} transparent opacity={opacity} />
    </line>
  );
}

function NeuralNetwork() {
  const groupRef = useRef();

  // Generate layers
  const layers = useMemo(() => {
    const result = [];
    const layerSizes = [6, 10, 12, 10, 8, 4, 3]; // Neural network structure
    const layerColors = ['#00d4ff', '#00d4ff', '#b44aff', '#b44aff', '#ff2d95', '#ff2d95', '#00ff88'];
    const spacing = 1.2;

    layerSizes.forEach((size, layerIdx) => {
      const nodes = [];
      const xPos = (layerIdx - (layerSizes.length - 1) / 2) * spacing;

      for (let i = 0; i < size; i++) {
        const yPos = (i - (size - 1) / 2) * 0.35;
        const zPos = (Math.random() - 0.5) * 0.3;
        nodes.push({
          position: [xPos, yPos, zPos],
          color: layerColors[layerIdx],
        });
      }
      result.push(nodes);
    });
    return result;
  }, []);

  // Generate connections between adjacent layers
  const connections = useMemo(() => {
    const conns = [];
    for (let l = 0; l < layers.length - 1; l++) {
      const curr = layers[l];
      const next = layers[l + 1];
      for (let i = 0; i < curr.length; i++) {
        // Connect to a subset of next layer nodes
        const connectCount = Math.min(3, next.length);
        for (let j = 0; j < connectCount; j++) {
          const targetIdx = Math.floor((j / connectCount) * next.length + i) % next.length;
          conns.push({
            start: curr[i].position,
            end: next[targetIdx].position,
            color: curr[i].color,
          });
        }
      }
    }
    return conns;
  }, [layers]);

  useFrame(({ clock }) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(clock.elapsedTime * 0.3) * 0.3;
      groupRef.current.rotation.x = Math.cos(clock.elapsedTime * 0.2) * 0.1;
    }
  });

  return (
    <group ref={groupRef}>
      {layers.flat().map((node, i) => (
        <NeuralNode key={`node-${i}`} {...node} />
      ))}
      {connections.map((conn, i) => (
        <NeuralConnection key={`conn-${i}`} {...conn} />
      ))}
    </group>
  );
}

export default function NeuralNetworkViz() {
  return (
    <div className="glass-card overflow-hidden relative" style={{ height: '200px' }}>
      <div className="absolute top-3 left-4 z-10 flex items-center gap-2">
        <span className="w-2 h-2 rounded-full bg-neon-blue animate-pulse" />
        <span className="text-[10px] font-display text-gray-400 uppercase tracking-wider">
          Neural Architecture
        </span>
      </div>
      <Canvas
        camera={{ position: [0, 0, 5], fov: 50 }}
        style={{ background: 'transparent' }}
      >
        <ambientLight intensity={0.3} />
        <pointLight position={[5, 5, 5]} intensity={0.8} color="#00d4ff" />
        <pointLight position={[-5, -5, 5]} intensity={0.5} color="#b44aff" />
        <NeuralNetwork />
        <OrbitControls
          enableZoom={false}
          enablePan={false}
          autoRotate
          autoRotateSpeed={0.5}
        />
      </Canvas>
    </div>
  );
}
