import React, { useState } from 'react';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { CreativeLandingPage } from './components/pages/CreativeLandingPage';
import { LandingPage } from './components/pages/LandingPage';
import { DashboardPage } from './components/pages/DashboardPage';
import { ReportPage } from './components/pages/ReportPage';
import { AboutPage } from './components/pages/AboutPage';
import { ComponentShowcase } from './components/pages/ComponentShowcase';

type Page = 'landing' | 'dashboard' | 'report' | 'about' | 'showcase';

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('landing');
  const [reportData, setReportData] = useState<any>(null);

  const handleNavigate = (page: string, data?: any) => {
    if (data) {
      setReportData(data);
    }
    setCurrentPage(page as Page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Use creative landing page without header/footer for cinematic experience
  if (currentPage === 'landing') {
    return <CreativeLandingPage onNavigate={handleNavigate} />;
  }

  return (
    <div className="min-h-screen flex flex-col">
      <Header onNavigate={handleNavigate} currentPage={currentPage} />
      
      {currentPage === 'dashboard' && <DashboardPage onNavigate={handleNavigate} />}
      {currentPage === 'report' && <ReportPage onNavigate={handleNavigate} reportData={reportData} />}
      {currentPage === 'about' && <AboutPage />}
      {currentPage === 'showcase' && <ComponentShowcase />}
      
      <Footer onNavigate={handleNavigate} />
    </div>
  );
}
