import React from 'react';
import { motion } from 'motion/react';
import astronautImage from 'figma:asset/868b782cadc788c733928c2597034af2ebdd4d87.png';

interface AstronautMascotProps {
  size?: number;
  className?: string;
}

export function AstronautMascot({ size = 150, className = '' }: AstronautMascotProps) {
  return (
    <motion.div
      className={`relative inline-block cursor-pointer ${className}`}
      style={{ width: size, height: size }}
      initial={{ opacity: 0, y: 20 }}
      animate={{ 
        opacity: 1, 
        y: 0,
      }}
      whileHover={{ 
        scale: 1.15,
        rotate: [0, -5, 5, -5, 0],
        transition: { 
          scale: { duration: 0.3 },
          rotate: { duration: 0.5, repeat: Infinity, repeatType: 'loop' }
        }
      }}
      whileTap={{ scale: 0.95 }}
      transition={{ duration: 0.5 }}
    >
      {/* Floating animation wrapper */}
      <motion.div
        animate={{
          y: [0, -10, 0],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "easeInOut"
        }}
      >
        {/* Glow effect on hover */}
        <motion.div
          className="absolute inset-0 rounded-full blur-xl"
          style={{
            background: 'radial-gradient(circle, rgba(108, 92, 231, 0.4), transparent 70%)',
          }}
          initial={{ opacity: 0 }}
          whileHover={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        />
        
        {/* Rocket trail effect */}
        <motion.div
          className="absolute bottom-0 left-1/2 -translate-x-1/2"
          style={{
            width: '40%',
            height: '60%',
            background: 'linear-gradient(to bottom, rgba(255, 107, 107, 0.6), rgba(255, 199, 95, 0.4), transparent)',
            filter: 'blur(8px)',
            borderRadius: '50%',
            zIndex: -1,
          }}
          initial={{ opacity: 0, scaleY: 0 }}
          whileHover={{ 
            opacity: [0.6, 0.8, 0.6],
            scaleY: [1, 1.3, 1],
          }}
          transition={{
            duration: 0.5,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />

        {/* Main image with background removal techniques */}
        <motion.img
          src={astronautImage}
          alt="Astronaut riding a rocket"
          className="w-full h-full object-contain"
          style={{
            filter: 'drop-shadow(0 4px 20px rgba(108, 92, 231, 0.3))',
            mixBlendMode: 'normal',
          }}
          whileHover={{
            filter: [
              'drop-shadow(0 4px 20px rgba(108, 92, 231, 0.3))',
              'drop-shadow(0 6px 30px rgba(0, 206, 201, 0.5))',
              'drop-shadow(0 4px 20px rgba(108, 92, 231, 0.3))',
            ],
          }}
          transition={{ duration: 1, repeat: Infinity }}
        />

        {/* Sparkle effects */}
        {[...Array(3)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute rounded-full bg-white"
            style={{
              width: 4,
              height: 4,
              top: `${20 + i * 25}%`,
              left: `${10 + i * 30}%`,
            }}
            initial={{ opacity: 0, scale: 0 }}
            whileHover={{
              opacity: [0, 1, 0],
              scale: [0, 1, 0],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              delay: i * 0.2,
            }}
          />
        ))}
      </motion.div>

      {/* Circular orbit ring on hover */}
      <motion.div
        className="absolute inset-0 rounded-full border-2"
        style={{
          borderColor: 'rgba(0, 206, 201, 0.5)',
          borderStyle: 'dashed',
        }}
        initial={{ scale: 0.8, opacity: 0, rotate: 0 }}
        whileHover={{ 
          scale: 1.3, 
          opacity: 0.6,
          rotate: 360,
        }}
        transition={{ 
          scale: { duration: 0.4 },
          opacity: { duration: 0.4 },
          rotate: { duration: 8, repeat: Infinity, ease: "linear" }
        }}
      />
    </motion.div>
  );
}
