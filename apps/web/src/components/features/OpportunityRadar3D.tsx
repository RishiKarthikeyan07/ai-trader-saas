'use client';

import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';

interface OpportunityPoint {
  symbol: string;
  confidence: number;
  riskGrade: 'LOW' | 'MEDIUM' | 'HIGH';
  expectedReturn: number;
}

interface OpportunityRadar3DProps {
  opportunities: OpportunityPoint[];
}

function OpportunityMesh({ opportunities }: { opportunities: OpportunityPoint[] }) {
  const groupRef = useRef<THREE.Group>(null);

  // Rotate the entire group slowly
  useFrame((state, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.y += delta * 0.1;
    }
  });

  const points = useMemo(() => {
    return opportunities.map((opp, idx) => {
      // Position based on confidence (radius) and expected return (height)
      const angle = (idx / opportunities.length) * Math.PI * 2;
      const radius = (1 - opp.confidence) * 3 + 1; // Lower confidence = further from center
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      const y = (opp.expectedReturn / 10) - 1; // Height based on expected return

      // Color based on risk grade
      const color =
        opp.riskGrade === 'LOW'
          ? '#10b981'
          : opp.riskGrade === 'MEDIUM'
            ? '#f59e0b'
            : '#ef4444';

      return { position: [x, y, z] as [number, number, number], color, symbol: opp.symbol };
    });
  }, [opportunities]);

  return (
    <group ref={groupRef}>
      {/* Central axis */}
      <mesh position={[0, 0, 0]}>
        <cylinderGeometry args={[0.05, 0.05, 6, 16]} />
        <meshStandardMaterial color="#06b6d4" opacity={0.3} transparent />
      </mesh>

      {/* Grid circles at different heights */}
      {[-2, 0, 2].map((y) => (
        <mesh key={y} position={[0, y, 0]} rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[3, 0.02, 16, 64]} />
          <meshStandardMaterial color="#3a3a3f" opacity={0.2} transparent />
        </mesh>
      ))}

      {/* Opportunity points */}
      {points.map((point, idx) => (
        <group key={idx} position={point.position}>
          <mesh>
            <sphereGeometry args={[0.15, 16, 16]} />
            <meshStandardMaterial
              color={point.color}
              emissive={point.color}
              emissiveIntensity={0.5}
            />
          </mesh>
          <Text
            position={[0, 0.4, 0]}
            fontSize={0.2}
            color="#ffffff"
            anchorX="center"
            anchorY="middle"
          >
            {point.symbol}
          </Text>
        </group>
      ))}
    </group>
  );
}

export function OpportunityRadar3D({ opportunities }: OpportunityRadar3DProps) {
  return (
    <div className="w-full h-[400px] rounded-xl overflow-hidden bg-graphite-900/50">
      <Canvas camera={{ position: [5, 3, 5], fov: 50 }}>
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        <OpportunityMesh opportunities={opportunities} />
        <OrbitControls
          enablePan={false}
          minDistance={3}
          maxDistance={15}
          autoRotate
          autoRotateSpeed={0.5}
        />
      </Canvas>
    </div>
  );
}
