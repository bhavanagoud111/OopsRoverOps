import React from 'react';
import { Card } from '../Card';
import { Rocket, Users, Code, Database, Brain, Shield, MapPin, FileText } from 'lucide-react';

export function AboutPage() {
  return (
    <main className="flex-1">
      <div className="max-w-[1440px] mx-auto px-6 py-8">
        <header className="text-center mb-12">
          <h1 className="mb-4">About OopsRoverOps</h1>
          <p className="text-[var(--color-text-secondary)] max-w-2xl mx-auto">
            An AI-powered space mission simulator combining autonomous agent orchestration 
            with real NASA Mars rover data.
          </p>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
          {/* Project Overview */}
          <section aria-labelledby="overview-heading">
            <Card>
              <div className="flex items-center gap-3 mb-4">
                <Rocket className="text-[var(--color-primary)]" size={32} aria-hidden="true" />
                <h2 id="overview-heading">Project Overview</h2>
              </div>
              <p className="text-[var(--color-text-secondary)] mb-4">
                OopsRoverOps simulates autonomous space rover missions using a multi-agent 
                AI system. Each mission involves four specialized agents working together:
              </p>
              <ul className="space-y-3" role="list">
                <li className="flex items-start gap-3">
                  <Brain className="text-[var(--color-primary)] flex-shrink-0 mt-1" size={20} aria-hidden="true" />
                  <div>
                    <strong>Planner Agent</strong>
                    <p className="text-[var(--color-text-secondary)]">
                      <small>Analyzes mission goals and calculates optimal routes</small>
                    </p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <MapPin className="text-[var(--color-secondary)] flex-shrink-0 mt-1" size={20} aria-hidden="true" />
                  <div>
                    <strong>Navigator Agent</strong>
                    <p className="text-[var(--color-text-secondary)]">
                      <small>Executes navigation and controls rover movement</small>
                    </p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <Shield className="text-[var(--color-accent)] flex-shrink-0 mt-1" size={20} aria-hidden="true" />
                  <div>
                    <strong>Safety Agent</strong>
                    <p className="text-[var(--color-text-secondary)]">
                      <small>Monitors systems and performs safety diagnostics</small>
                    </p>
                  </div>
                </li>
                <li className="flex items-start gap-3">
                  <FileText className="text-[var(--color-success)] flex-shrink-0 mt-1" size={20} aria-hidden="true" />
                  <div>
                    <strong>Reporter Agent</strong>
                    <p className="text-[var(--color-text-secondary)]">
                      <small>Collects data and generates mission reports</small>
                    </p>
                  </div>
                </li>
              </ul>
            </Card>
          </section>

          {/* Credits & APIs */}
          <section aria-labelledby="credits-heading">
            <Card>
              <div className="flex items-center gap-3 mb-4">
                <Database className="text-[var(--color-secondary)]" size={32} aria-hidden="true" />
                <h2 id="credits-heading">Credits & APIs</h2>
              </div>
              <div className="space-y-4">
                <div>
                  <h3 className="mb-2">NASA API Integration</h3>
                  <p className="text-[var(--color-text-secondary)] mb-2">
                    <small>
                      This application uses the NASA Mars Rover Photos API to fetch real imagery 
                      from Mars rovers including Curiosity, Opportunity, and Spirit.
                    </small>
                  </p>
                  <a
                    href="https://api.nasa.gov"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-[var(--color-primary)] hover:underline focus:outline-none focus:ring-2 focus:ring-[var(--color-primary)] rounded"
                  >
                    <small>Get your NASA API key →</small>
                  </a>
                </div>

                <div>
                  <h3 className="mb-2">Open Source Attribution</h3>
                  <p className="text-[var(--color-text-secondary)]">
                    <small>
                      Images and data courtesy of NASA/JPL-Caltech. This project is built for 
                      educational purposes and is not affiliated with NASA.
                    </small>
                  </p>
                </div>
              </div>
            </Card>
          </section>
        </div>

        {/* Tech Stack */}
        <section aria-labelledby="tech-heading" className="mb-12">
          <Card>
            <div className="flex items-center gap-3 mb-6">
              <Code className="text-[var(--color-accent)]" size={32} aria-hidden="true" />
              <h2 id="tech-heading">Technology Stack</h2>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <h3 className="mb-2">Frontend</h3>
                <ul className="text-[var(--color-text-secondary)] space-y-1" role="list">
                  <li><small>React + TypeScript</small></li>
                  <li><small>Tailwind CSS</small></li>
                  <li><small>HTML5 Canvas</small></li>
                  <li><small>Lucide Icons</small></li>
                </ul>
              </div>
              <div>
                <h3 className="mb-2">Features</h3>
                <ul className="text-[var(--color-text-secondary)] space-y-1" role="list">
                  <li><small>Real-time simulation</small></li>
                  <li><small>Agent orchestration</small></li>
                  <li><small>NASA API integration</small></li>
                  <li><small>WCAG AA accessible</small></li>
                </ul>
              </div>
              <div>
                <h3 className="mb-2">Architecture</h3>
                <ul className="text-[var(--color-text-secondary)] space-y-1" role="list">
                  <li><small>Component-based</small></li>
                  <li><small>Responsive design</small></li>
                  <li><small>Semantic HTML</small></li>
                  <li><small>Clean code practices</small></li>
                </ul>
              </div>
            </div>
          </Card>
        </section>

        {/* Team */}
        <section aria-labelledby="team-heading">
          <Card>
            <div className="flex items-center gap-3 mb-6">
              <Users className="text-[var(--color-success)]" size={32} aria-hidden="true" />
              <h2 id="team-heading">Project Purpose</h2>
            </div>
            <p className="text-[var(--color-text-secondary)] max-w-3xl">
              OopsRoverOps was built as a learning project to explore autonomous agent systems, 
              real-time data visualization, and integration with external APIs. It demonstrates how 
              multiple AI agents can coordinate to accomplish complex tasks, using space exploration 
              as an engaging and educational context. The project emphasizes clean code architecture, 
              accessibility standards, and user experience design.
            </p>
          </Card>
        </section>
      </div>
    </main>
  );
}