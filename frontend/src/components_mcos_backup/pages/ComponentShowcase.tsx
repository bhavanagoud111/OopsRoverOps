import React, { useState } from 'react';
import { Card } from '../Card';
import { Button } from '../Button';
import { RoverAnimated } from '../animated/RoverAnimated';
import { CompanionBot } from '../animated/CompanionBot';
import { DestinationMarker } from '../animated/DestinationMarker';
import { SpeechBubble } from '../animated/SpeechBubble';
import { AstronautMascot } from '../animated/AstronautMascot';
import { RoverSVG } from '../svg/RoverSVG';
import { SpaceshipSVG } from '../svg/SpaceshipSVG';
import { CompanionSVG } from '../svg/CompanionSVG';

export function ComponentShowcase() {
  const [roverState, setRoverState] = useState<'idle' | 'moving' | 'reachedTarget'>('idle');
  const [companionState, setCompanionState] = useState<'idle' | 'talking' | 'excited'>('idle');
  const [markerVariant, setMarkerVariant] = useState<'default' | 'glow'>('default');

  return (
    <main className="flex-1">
      <div className="max-w-[1440px] mx-auto px-6 py-8">
        <header className="mb-8">
          <h1 className="mb-4">Component Showcase</h1>
          <p className="text-[var(--color-text-secondary)]">
            Interactive demonstration of all OopsRoverOps animated components and SVG exports
          </p>
        </header>

        {/* Hero Mascot Section */}
        <section className="mb-12" aria-labelledby="mascot-heading">
          <h2 id="mascot-heading" className="mb-6">Astronaut Mascot</h2>
          <Card>
            <div className="flex flex-col items-center py-8">
              <h3 className="mb-6">Interactive Astronaut Hero</h3>
              <div className="flex justify-center items-center h-64 mb-4">
                <AstronautMascot size={200} />
              </div>
              <p className="text-[var(--color-text-secondary)] text-center max-w-md">
                <small>
                  Hover over the astronaut to see animations: floating motion, glow effects, 
                  rocket trail, sparkles, wobble rotation, and orbital ring. Click to see tap effect!
                </small>
              </p>
            </div>
          </Card>
        </section>

        {/* Animated Components Section */}
        <section className="mb-12" aria-labelledby="animated-heading">
          <h2 id="animated-heading" className="mb-6">Animated Components</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {/* Rover Animated */}
            <Card>
              <h3 className="mb-4">Rover / Animated</h3>
              <div className="flex justify-center items-center h-32 mb-4 bg-[var(--color-background)] rounded-lg">
                <RoverAnimated state={roverState} size={80} />
              </div>
              <div className="space-y-2">
                <p className="text-[var(--color-text-secondary)] mb-2">
                  <small>Current state: <strong>{roverState}</strong></small>
                </p>
                <div className="flex gap-2">
                  <Button
                    onClick={() => setRoverState('idle')}
                    variant={roverState === 'idle' ? 'primary' : 'outline'}
                    size="sm"
                  >
                    Idle
                  </Button>
                  <Button
                    onClick={() => setRoverState('moving')}
                    variant={roverState === 'moving' ? 'primary' : 'outline'}
                    size="sm"
                  >
                    Moving
                  </Button>
                  <Button
                    onClick={() => setRoverState('reachedTarget')}
                    variant={roverState === 'reachedTarget' ? 'primary' : 'outline'}
                    size="sm"
                  >
                    Target
                  </Button>
                </div>
              </div>
            </Card>

            {/* Companion Bot */}
            <Card>
              <h3 className="mb-4">CompanionBot / Avatar</h3>
              <div className="flex justify-center items-center h-32 mb-4 bg-[var(--color-background)] rounded-lg">
                <CompanionBot 
                  state={companionState} 
                  message={companionState === 'talking' ? "I'm talking!" : companionState === 'excited' ? "Yay!" : ""}
                  size={80} 
                />
              </div>
              <div className="space-y-2">
                <p className="text-[var(--color-text-secondary)] mb-2">
                  <small>Current state: <strong>{companionState}</strong></small>
                </p>
                <div className="flex gap-2">
                  <Button
                    onClick={() => setCompanionState('idle')}
                    variant={companionState === 'idle' ? 'primary' : 'outline'}
                    size="sm"
                  >
                    Idle
                  </Button>
                  <Button
                    onClick={() => setCompanionState('talking')}
                    variant={companionState === 'talking' ? 'primary' : 'outline'}
                    size="sm"
                  >
                    Talking
                  </Button>
                  <Button
                    onClick={() => setCompanionState('excited')}
                    variant={companionState === 'excited' ? 'primary' : 'outline'}
                    size="sm"
                  >
                    Excited
                  </Button>
                </div>
              </div>
            </Card>

            {/* Destination Marker */}
            <Card>
              <h3 className="mb-4">DestinationMarker / Spaceship</h3>
              <div className="flex justify-center items-center h-32 mb-4 bg-[var(--color-background)] rounded-lg">
                <DestinationMarker variant={markerVariant} size={60} />
              </div>
              <div className="space-y-2">
                <p className="text-[var(--color-text-secondary)] mb-2">
                  <small>Current variant: <strong>{markerVariant}</strong></small>
                </p>
                <div className="flex gap-2">
                  <Button
                    onClick={() => setMarkerVariant('default')}
                    variant={markerVariant === 'default' ? 'primary' : 'outline'}
                    size="sm"
                  >
                    Default
                  </Button>
                  <Button
                    onClick={() => setMarkerVariant('glow')}
                    variant={markerVariant === 'glow' ? 'primary' : 'outline'}
                    size="sm"
                  >
                    Glow
                  </Button>
                </div>
              </div>
            </Card>
          </div>

          {/* Speech Bubble */}
          <Card>
            <h3 className="mb-4">SpeechBubble Component</h3>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="flex flex-col items-center gap-4">
                <p className="text-[var(--color-text-secondary)]"><small>Left</small></p>
                <div className="bg-[var(--color-background)] p-8 rounded-lg">
                  <SpeechBubble message="Hello from left!" variant="left" />
                </div>
              </div>
              <div className="flex flex-col items-center gap-4">
                <p className="text-[var(--color-text-secondary)]"><small>Right</small></p>
                <div className="bg-[var(--color-background)] p-8 rounded-lg">
                  <SpeechBubble message="Hello from right!" variant="right" />
                </div>
              </div>
              <div className="flex flex-col items-center gap-4">
                <p className="text-[var(--color-text-secondary)]"><small>Top</small></p>
                <div className="bg-[var(--color-background)] p-8 rounded-lg">
                  <SpeechBubble message="Hello from top!" variant="top" />
                </div>
              </div>
              <div className="flex flex-col items-center gap-4">
                <p className="text-[var(--color-text-secondary)]"><small>Bottom</small></p>
                <div className="bg-[var(--color-background)] p-8 rounded-lg">
                  <SpeechBubble message="Hello from bottom!" variant="bottom" />
                </div>
              </div>
            </div>
          </Card>
        </section>

        {/* SVG Exports Section */}
        <section aria-labelledby="svg-heading">
          <h2 id="svg-heading" className="mb-6">SVG Exports</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <Card>
              <h3 className="mb-4">Rover SVG</h3>
              <div className="flex justify-center items-center h-32 bg-[var(--color-background)] rounded-lg">
                <RoverSVG size={80} />
              </div>
              <p className="mt-4 text-[var(--color-text-secondary)]">
                <small>Pure SVG export for custom integration</small>
              </p>
            </Card>

            <Card>
              <h3 className="mb-4">Spaceship SVG</h3>
              <div className="flex justify-center items-center h-32 bg-[var(--color-background)] rounded-lg">
                <SpaceshipSVG size={60} variant="glow" />
              </div>
              <p className="mt-4 text-[var(--color-text-secondary)]">
                <small>Purple gradient spaceship with glow option</small>
              </p>
            </Card>

            <Card>
              <h3 className="mb-4">Companion SVG</h3>
              <div className="flex justify-center items-center h-32 bg-[var(--color-background)] rounded-lg">
                <CompanionSVG size={80} eyeState="happy" />
              </div>
              <p className="mt-4 text-[var(--color-text-secondary)]">
                <small>AI companion with eye state variants</small>
              </p>
            </Card>
          </div>
        </section>

        {/* Component Info */}
        <section className="mt-12" aria-labelledby="info-heading">
          <Card>
            <h2 id="info-heading" className="mb-4">Component Information</h2>
            <div className="space-y-4 text-[var(--color-text-secondary)]">
              <p>
                All components are built following engineering best practices:
              </p>
              <ul className="list-disc list-inside space-y-2 ml-4">
                <li>WCAG AA accessibility compliance</li>
                <li>Reusable component architecture</li>
                <li>TypeScript interfaces for type safety</li>
                <li>Flexbox/grid layouts (no absolute positioning)</li>
                <li>Pure code components (not flattened images)</li>
                <li>Animation states ready for GSAP integration</li>
              </ul>
              <p className="mt-4">
                <strong>See COMPONENT_GUIDE.md</strong> for detailed documentation on props, variants, and GSAP integration examples.
              </p>
            </div>
          </Card>
        </section>
      </div>
    </main>
  );
}