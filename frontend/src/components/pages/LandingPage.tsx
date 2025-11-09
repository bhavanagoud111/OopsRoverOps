import React from 'react';
import { Button } from '../Button';
import { Card } from '../Card';
import { Rocket, Brain, MapPin, Shield, FileText } from 'lucide-react';
import { AstronautMascot } from '../animated/AstronautMascot';

interface LandingPageProps {
  onNavigate: (page: string) => void;
}

export function LandingPage({ onNavigate }: LandingPageProps) {
  return (
    <main className="flex-1">
      <div className="max-w-[1440px] mx-auto px-6 py-16">
        {/* Hero Section */}
        <section className="text-center mb-16 relative" aria-labelledby="hero-heading">
          {/* Astronaut Mascot */}
          <div className="flex justify-center mb-8">
            <AstronautMascot size={200} />
          </div>
          
          <h1 id="hero-heading" className="mb-6">
            Welcome to OopsRoverOps
          </h1>
          <p className="text-[var(--color-text-secondary)] max-w-2xl mx-auto mb-8">
            An AI-powered space mission simulator combining autonomous agent orchestration with real NASA data.
            Watch as four specialized AI agents work together to plan and execute Mars rover missions in real-time.
          </p>
          <Button
            onClick={() => onNavigate('dashboard')}
            variant="primary"
            size="lg"
            aria-label="Start your mission"
          >
            <Rocket size={20} aria-hidden="true" />
            Start Mission
          </Button>
        </section>

        {/* Features Section */}
        <section aria-labelledby="features-heading">
          <h2 id="features-heading" className="text-center mb-12">
            Mission Capabilities
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card>
              <div className="flex flex-col items-center text-center">
                <Brain size={40} className="text-[var(--color-primary)] mb-4" aria-hidden="true" />
                <h3 className="mb-2">AI Agent Planner</h3>
                <p className="text-[var(--color-text-secondary)]">
                  <small>Intelligent mission planning and route optimization</small>
                </p>
              </div>
            </Card>

            <Card>
              <div className="flex flex-col items-center text-center">
                <MapPin size={40} className="text-[var(--color-secondary)] mb-4" aria-hidden="true" />
                <h3 className="mb-2">Real-time Navigation</h3>
                <p className="text-[var(--color-text-secondary)]">
                  <small>Live rover movement visualization on 2D terrain</small>
                </p>
              </div>
            </Card>

            <Card>
              <div className="flex flex-col items-center text-center">
                <Shield size={40} className="text-[var(--color-accent)] mb-4" aria-hidden="true" />
                <h3 className="mb-2">Safety Monitoring</h3>
                <p className="text-[var(--color-text-secondary)]">
                  <small>Continuous safety checks and diagnostics</small>
                </p>
              </div>
            </Card>

            <Card>
              <div className="flex flex-col items-center text-center">
                <FileText size={40} className="text-[var(--color-success)] mb-4" aria-hidden="true" />
                <h3 className="mb-2">Mission Reports</h3>
                <p className="text-[var(--color-text-secondary)]">
                  <small>Detailed mission summaries with NASA imagery</small>
                </p>
              </div>
            </Card>
          </div>
        </section>
      </div>
    </main>
  );
}