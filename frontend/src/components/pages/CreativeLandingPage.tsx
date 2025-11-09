import React, { useEffect, useRef, useState } from 'react';
import { Brain, MapPin, Shield, Rocket } from 'lucide-react';
import { motion } from 'motion/react';
import spaceRoverBg from 'figma:asset/3d02b908cd238433f2979cfd7e5e716f8472ec41.png';

interface CreativeLandingPageProps {
  onNavigate: (page: string) => void;
}

export function CreativeLandingPage({ onNavigate }: CreativeLandingPageProps) {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return;
      const { clientX, clientY } = e;
      const { innerWidth, innerHeight } = window;
      
      // Normalize to -1 to 1 range
      const x = (clientX / innerWidth - 0.5) * 2;
      const y = (clientY / innerHeight - 0.5) * 2;
      
      setMousePosition({ x, y });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div
      ref={containerRef}
      style={{
        width: '100vw',
        height: '100vh',
        overflow: 'auto',
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#000000',
      }}
    >
      {/* Background Image - Fixed and Centered */}
      <div
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          zIndex: 0,
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            backgroundImage: `url(${spaceRoverBg})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center center',
            backgroundRepeat: 'no-repeat',
          }}
        />
        
        {/* Dark Gradient Overlay */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            background: 'linear-gradient(180deg, rgba(0, 0, 0, 0.6) 0%, rgba(0, 0, 0, 0.7) 50%, rgba(10, 14, 39, 0.85) 100%)',
            pointerEvents: 'none',
          }}
        />

        {/* Floating Stars */}
        {[...Array(40)].map((_, i) => (
          <motion.div
            key={i}
            style={{
              position: 'absolute',
              width: Math.random() * 3 + 1 + 'px',
              height: Math.random() * 3 + 1 + 'px',
              borderRadius: '50%',
              background: '#ffffff',
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
              boxShadow: '0 0 4px rgba(255, 255, 255, 0.8)',
            }}
            animate={{
              opacity: [0.2, 1, 0.2],
              scale: [1, 1.5, 1],
            }}
            transition={{
              duration: Math.random() * 3 + 2,
              repeat: Infinity,
              ease: 'easeInOut',
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>

      {/* Planet 1 - Saturn (Top Left) - Yellow/Beige with Rings */}
      <motion.div
        style={{
          position: 'absolute',
          top: '8%',
          left: '12%',
          width: '160px',
          height: '160px',
          zIndex: 1,
        }}
        animate={{
          y: [0, -25, 0],
          x: [0, mousePosition.x * -30, 0],
          scale: [1, 1.05, 1],
        }}
        transition={{
          y: { duration: 6, repeat: Infinity, ease: 'easeInOut' },
          x: { duration: 8, repeat: Infinity, ease: 'easeInOut' },
          scale: { duration: 5, repeat: Infinity, ease: 'easeInOut' },
        }}
      >
        {/* Saturn body */}
        <div
          style={{
            width: '100%',
            height: '100%',
            borderRadius: '50%',
            background: 'radial-gradient(circle at 35% 35%, #F4E4C1, #E6D5A8, #D4C19C)',
            boxShadow: '0 0 60px rgba(230, 213, 168, 0.6), inset -25px -25px 50px rgba(0, 0, 0, 0.4)',
          }}
        />
        
        {/* Saturn rings - positioned absolutely within the container */}
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            width: '180%',
            height: '35%',
            border: '3px solid rgba(230, 213, 168, 0.5)',
            borderRadius: '50%',
            transform: 'translate(-50%, -50%) rotateX(75deg)',
            boxShadow: 'inset 0 0 20px rgba(0, 0, 0, 0.3)',
            pointerEvents: 'none',
          }}
        />
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            width: '150%',
            height: '28%',
            border: '2px solid rgba(200, 180, 140, 0.4)',
            borderRadius: '50%',
            transform: 'translate(-50%, -50%) rotateX(75deg)',
            pointerEvents: 'none',
          }}
        />
      </motion.div>

      {/* Planet 2 - Jupiter (Bottom Left) - Orange/Brown with bands */}
      <motion.div
        style={{
          position: 'absolute',
          bottom: '18%',
          left: '8%',
          width: '200px',
          height: '200px',
          borderRadius: '50%',
          background: 'radial-gradient(circle at 30% 30%, #E8B88B, #D4A574, #C89664, #B87333)',
          boxShadow: '0 0 70px rgba(200, 150, 100, 0.6), inset -30px -30px 60px rgba(0, 0, 0, 0.5)',
          zIndex: 1,
        }}
        animate={{
          y: [0, 20, 0],
          x: [0, mousePosition.x * 25, 0],
          rotate: [0, -360],
          scale: [1, 1.08, 1],
        }}
        transition={{
          y: { duration: 7, repeat: Infinity, ease: 'easeInOut' },
          x: { duration: 9, repeat: Infinity, ease: 'easeInOut' },
          rotate: { duration: 50, repeat: Infinity, ease: 'linear' },
          scale: { duration: 6, repeat: Infinity, ease: 'easeInOut' },
        }}
      >
        {/* Jupiter bands/stripes */}
        <div style={{
          position: 'absolute',
          width: '100%',
          height: '15%',
          top: '35%',
          left: 0,
          background: 'rgba(160, 100, 60, 0.3)',
          borderRadius: '50%',
        }} />
        <div style={{
          position: 'absolute',
          width: '100%',
          height: '12%',
          top: '55%',
          left: 0,
          background: 'rgba(140, 90, 50, 0.25)',
          borderRadius: '50%',
        }} />
        {/* Great Red Spot */}
        <div style={{
          position: 'absolute',
          width: '25%',
          height: '18%',
          top: '45%',
          right: '30%',
          background: 'radial-gradient(circle, rgba(180, 80, 60, 0.6), rgba(160, 70, 50, 0.4))',
          borderRadius: '50%',
        }} />
      </motion.div>

      {/* Planet 3 - Earth (Top Right) - Blue/Green */}
      <motion.div
        style={{
          position: 'absolute',
          top: '5%',
          right: '5%',
          width: '240px',
          height: '240px',
          borderRadius: '50%',
          background: 'radial-gradient(circle at 30% 30%, #5B9BD5, #4A8AC2, #3A7AA8, #2A5A7A)',
          boxShadow: '0 0 80px rgba(90, 155, 213, 0.6), inset -35px -35px 70px rgba(0, 0, 0, 0.4)',
          zIndex: 1,
        }}
        animate={{
          y: [0, -30, 0],
          x: [0, mousePosition.x * -35, 0],
          rotate: [0, 360],
          scale: [1, 1.1, 1],
        }}
        transition={{
          y: { duration: 8, repeat: Infinity, ease: 'easeInOut' },
          x: { duration: 10, repeat: Infinity, ease: 'easeInOut' },
          rotate: { duration: 60, repeat: Infinity, ease: 'linear' },
          scale: { duration: 7, repeat: Infinity, ease: 'easeInOut' },
        }}
      >
        {/* Continents/land masses */}
        <div style={{
          position: 'absolute',
          width: '35%',
          height: '30%',
          top: '25%',
          left: '15%',
          background: 'rgba(76, 153, 76, 0.7)',
          borderRadius: '40% 60% 50% 50%',
          filter: 'blur(2px)',
        }} />
        <div style={{
          position: 'absolute',
          width: '28%',
          height: '25%',
          top: '45%',
          right: '20%',
          background: 'rgba(90, 160, 80, 0.65)',
          borderRadius: '50% 40% 60% 50%',
          filter: 'blur(2px)',
        }} />
        {/* Atmosphere glow */}
        <motion.div
          style={{
            position: 'absolute',
            inset: -20,
            borderRadius: '50%',
            background: 'radial-gradient(circle, rgba(90, 155, 213, 0.3), transparent 70%)',
            filter: 'blur(20px)',
          }}
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.4, 0.7, 0.4],
          }}
          transition={{
            duration: 4,
            repeat: Infinity,
            ease: 'easeInOut',
          }}
        />
      </motion.div>

      {/* Planet 4 - Neptune (Bottom Right) - Deep Blue */}
      <motion.div
        style={{
          position: 'absolute',
          bottom: '8%',
          right: '8%',
          width: '180px',
          height: '180px',
          borderRadius: '50%',
          background: 'radial-gradient(circle at 35% 35%, #4A6FA5, #3A5A8A, #2A4A70, #1A3A5A)',
          boxShadow: '0 0 70px rgba(74, 111, 165, 0.6), inset -25px -25px 50px rgba(0, 0, 0, 0.4)',
          zIndex: 1,
        }}
        animate={{
          y: [0, -18, 0],
          x: [0, mousePosition.x * 20, 0],
          rotate: [0, -360],
          scale: [1, 1.15, 1],
        }}
        transition={{
          y: { duration: 5.5, repeat: Infinity, ease: 'easeInOut' },
          x: { duration: 7, repeat: Infinity, ease: 'easeInOut' },
          rotate: { duration: 45, repeat: Infinity, ease: 'linear' },
          scale: { duration: 4.5, repeat: Infinity, ease: 'easeInOut' },
        }}
      >
        {/* Neptune storms/bands */}
        <div style={{
          position: 'absolute',
          width: '40%',
          height: '35%',
          top: '30%',
          right: '25%',
          background: 'rgba(60, 100, 140, 0.4)',
          borderRadius: '50%',
          filter: 'blur(8px)',
        }} />
      </motion.div>

      {/* Planet 5 - Mars (Middle Left) - Red/Orange */}
      <motion.div
        style={{
          position: 'absolute',
          top: '32%',
          left: '18%',
          width: '130px',
          height: '130px',
          borderRadius: '50%',
          background: 'radial-gradient(circle at 40% 40%, #CD5C5C, #C04C4C, #B33C3C, #A52A2A)',
          boxShadow: '0 0 50px rgba(205, 92, 92, 0.5), inset -18px -18px 35px rgba(0, 0, 0, 0.4)',
          zIndex: 1,
        }}
        animate={{
          y: [0, 15, 0],
          x: [0, mousePosition.x * -15, 0],
          rotate: [0, 360],
          scale: [1, 1.1, 1],
        }}
        transition={{
          y: { duration: 4.5, repeat: Infinity, ease: 'easeInOut' },
          x: { duration: 6, repeat: Infinity, ease: 'easeInOut' },
          rotate: { duration: 35, repeat: Infinity, ease: 'linear' },
          scale: { duration: 4, repeat: Infinity, ease: 'easeInOut' },
        }}
      >
        {/* Mars surface features/craters */}
        <div style={{
          position: 'absolute',
          width: '20%',
          height: '20%',
          top: '25%',
          left: '30%',
          background: 'rgba(140, 50, 40, 0.5)',
          borderRadius: '50%',
        }} />
        <div style={{
          position: 'absolute',
          width: '15%',
          height: '15%',
          top: '55%',
          right: '25%',
          background: 'rgba(120, 40, 30, 0.4)',
          borderRadius: '50%',
        }} />
      </motion.div>

      {/* Main Content */}
      <motion.div
        style={{
          position: 'relative',
          zIndex: 5,
          textAlign: 'center',
          maxWidth: '1000px',
          width: '100%',
        }}
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: 'easeOut', delay: 0.5 }}
      >
        {/* Title */}
        <motion.h1
          style={{
            fontSize: 'clamp(48px, 7vw, 84px)',
            fontWeight: 700,
            background: 'linear-gradient(135deg, #00d9c0 0%, #00cec9 50%, #6b5ce7 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            backgroundClip: 'text',
            marginBottom: '16px',
            letterSpacing: '-0.02em',
          }}
          animate={{
            textShadow: [
              '0 0 80px rgba(0, 217, 192, 0.5)',
              '0 0 100px rgba(0, 217, 192, 0.8)',
              '0 0 80px rgba(0, 217, 192, 0.5)',
            ]
          }}
          transition={{
            duration: 3,
            repeat: Infinity,
            ease: 'easeInOut'
          }}
        >
          OopsRoverOps
        </motion.h1>

        {/* Subtitle */}
        <motion.p
          style={{
            fontSize: 'clamp(16px, 2vw, 20px)',
            color: 'var(--color-text-secondary)',
            marginBottom: '40px',
            lineHeight: 1.6,
          }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8, duration: 0.6 }}
        >
          Autonomous Goal-Based Rover Mission System
        </motion.p>

        {/* CTA Button */}
        <motion.button
          onClick={() => onNavigate('dashboard')}
          style={{
            padding: '16px 40px',
            fontSize: '17px',
            fontWeight: 600,
            color: '#ffffff',
            background: 'linear-gradient(135deg, #6b5ce7 0%, #00d9c0 100%)',
            border: 'none',
            borderRadius: '12px',
            cursor: 'pointer',
            boxShadow: '0 8px 32px rgba(107, 92, 231, 0.5)',
            position: 'relative',
            overflow: 'hidden',
          }}
          whileHover={{
            scale: 1.05,
            boxShadow: '0 12px 48px rgba(107, 92, 231, 0.7)',
          }}
          whileTap={{ scale: 0.98 }}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.0, duration: 0.5 }}
        >
          <Rocket
            size={20}
            style={{
              display: 'inline-block',
              marginRight: '10px',
              verticalAlign: 'middle',
            }}
          />
          Launch Mission Control
        </motion.button>

        {/* Subtext */}
        <motion.p
          style={{
            marginTop: '20px',
            fontSize: '12px',
            color: 'var(--color-text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '2px',
          }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2, duration: 0.6 }}
        >
          Runs Autonomous Agent Orchestration
        </motion.p>

        {/* Feature Cards */}
        <motion.div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(3, 1fr)',
            gap: '20px',
            marginTop: '60px',
            maxWidth: '850px',
            margin: '60px auto 0',
          }}
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.4, duration: 0.7 }}
        >
          {/* Feature 1 */}
          <motion.div
            style={{
              background: 'rgba(26, 31, 58, 0.7)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(107, 92, 231, 0.3)',
              borderRadius: '16px',
              padding: '28px 20px',
              textAlign: 'center',
            }}
            whileHover={{
              scale: 1.03,
              borderColor: 'rgba(107, 92, 231, 0.6)',
              boxShadow: '0 8px 32px rgba(107, 92, 231, 0.3)',
            }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <div
              style={{
                width: '52px',
                height: '52px',
                margin: '0 auto 14px',
                background: 'linear-gradient(135deg, #6b5ce7, #a47cff)',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Brain size={26} color="#ffffff" />
            </div>
            <h3 style={{ marginBottom: '10px', color: 'var(--color-text-primary)', fontSize: '17px' }}>
              AI Mission Planner
            </h3>
            <p style={{ fontSize: '13px', color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
              Converts mission goals into step-by-step executable plans
            </p>
          </motion.div>

          {/* Feature 2 */}
          <motion.div
            style={{
              background: 'rgba(26, 31, 58, 0.7)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(0, 217, 192, 0.3)',
              borderRadius: '16px',
              padding: '28px 20px',
              textAlign: 'center',
            }}
            whileHover={{
              scale: 1.03,
              borderColor: 'rgba(0, 217, 192, 0.6)',
              boxShadow: '0 8px 32px rgba(0, 217, 192, 0.3)',
            }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <div
              style={{
                width: '52px',
                height: '52px',
                margin: '0 auto 14px',
                background: 'linear-gradient(135deg, #00d9c0, #00cec9)',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <MapPin size={26} color="#ffffff" />
            </div>
            <h3 style={{ marginBottom: '10px', color: 'var(--color-text-primary)', fontSize: '17px' }}>
              Autonomous Navigation
            </h3>
            <p style={{ fontSize: '13px', color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
              Rover moves intelligently and avoids hazards — no human micromanagement
            </p>
          </motion.div>

          {/* Feature 3 */}
          <motion.div
            style={{
              background: 'rgba(26, 31, 58, 0.7)',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 61, 139, 0.3)',
              borderRadius: '16px',
              padding: '28px 20px',
              textAlign: 'center',
            }}
            whileHover={{
              scale: 1.03,
              borderColor: 'rgba(255, 61, 139, 0.6)',
              boxShadow: '0 8px 32px rgba(255, 61, 139, 0.3)',
            }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            <div
              style={{
                width: '52px',
                height: '52px',
                margin: '0 auto 14px',
                background: 'linear-gradient(135deg, #ff3d8b, #ff6b9d)',
                borderRadius: '12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <Shield size={26} color="#ffffff" />
            </div>
            <h3 style={{ marginBottom: '10px', color: 'var(--color-text-primary)', fontSize: '17px' }}>
              Safety & Diagnostics
            </h3>
            <p style={{ fontSize: '13px', color: 'var(--color-text-secondary)', lineHeight: 1.6 }}>
              Safety agent validates decisions and prevents rover damage
            </p>
          </motion.div>
        </motion.div>
      </motion.div>
    </div>
  );
}
