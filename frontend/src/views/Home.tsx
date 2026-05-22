/// <reference types="vite/client" />
import React, { Suspense, lazy } from 'react';
import { ArrowRight, ScanLine } from 'lucide-react';
import { Link } from 'react-router-dom';

const ImpactSection = lazy(() => import('../components/home/ImpactSection'));
const CategoriesSection = lazy(() => import('../components/home/CategoriesSection'));
const HowItWorksSection = lazy(() => import('../components/home/HowItWorksSection'));

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Hero Section */}
      <section className="relative w-full overflow-hidden bg-white dark:bg-[#121824] py-20 lg:py-32 transition-colors duration-300">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-blue-50 dark:from-emerald-900/20 via-white dark:via-[#121824] to-white dark:to-[#121824] transition-colors duration-300"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid lg:grid-cols-2 gap-12 lg:gap-8 items-center">
            
            {/* Left Column */}
            <div className="flex flex-col items-start text-left max-w-2xl">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-[#10B981]/30 bg-[#ECFDF5] dark:bg-[#10B981]/10 text-[#059669] dark:text-[#10B981] text-sm font-medium mb-8">
                <span>🌍</span> Vision: Towards a Green India
              </div>
              
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-extrabold text-[#0F172A] dark:text-[#F9FAFB] tracking-tight mb-6 font-display leading-[1.1] transition-colors duration-300">
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#10B981] to-teal-500 xl:whitespace-nowrap">AI-Powered</span> Waste Sorting for a Cleaner Tomorrow.
              </h1>
              
              <p className="text-lg text-slate-700 dark:text-slate-300 mb-10 leading-relaxed font-light transition-colors duration-300">
                Leveraging Deep Learning and Transfer Learning to automate garbage classification. Powering the circular economy, one pixel at a time.
              </p>
              
              <div className="flex flex-col sm:flex-row items-center gap-4 w-full sm:w-auto">
                <Link 
                  to="/classify"
                  className="w-full sm:w-auto inline-flex items-center justify-center gap-2 bg-[#10B981] hover:bg-[#059669] text-white px-8 py-4 rounded-xl font-bold text-base transition-all shadow-md hover:shadow-lg dark:text-[#121824] dark:shadow-[0_0_30px_-5px_rgba(16,185,129,0.4)] dark:hover:shadow-[0_0_40px_-5px_rgba(16,185,129,0.6)] hover:-translate-y-0.5 min-h-[48px]"
                >
                  Launch AI Classifier
                  <ArrowRight className="w-5 h-5" />
                </Link>
                <a 
                  href="#categories"
                  className="w-full sm:w-auto inline-flex items-center justify-center gap-2 border-2 border-gray-200 dark:border-slate-700 bg-white dark:bg-transparent hover:bg-gray-50 dark:hover:border-slate-500 text-[#0F172A] dark:text-slate-300 px-8 py-4 rounded-xl font-semibold text-base transition-all min-h-[48px]"
                >
                  Learn More
                </a>
              </div>
            </div>

            {/* Right Column */}
            <div className="relative w-full max-w-[280px] sm:max-w-md mx-auto lg:max-w-none lg:ml-auto perspective-[1000px]">
              <div className="relative rounded-2xl overflow-hidden border border-gray-200 dark:border-slate-700/50 bg-white/50 dark:bg-slate-800/30 backdrop-blur-xl shadow-xl dark:shadow-2xl transform lg:rotate-y-[-5deg] lg:rotate-x-[5deg] hover:rotate-0 transition-transform duration-700">
                <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-slate-700/50 bg-gray-50 dark:bg-slate-900/50">
                  <div className="flex gap-2">
                    <div className="w-3 h-3 rounded-full bg-red-400 dark:bg-red-500/80"></div>
                    <div className="w-3 h-3 rounded-full bg-yellow-400 dark:bg-yellow-500/80"></div>
                    <div className="w-3 h-3 rounded-full bg-green-400 dark:bg-green-500/80"></div>
                  </div>
                  <div className="text-[10px] sm:text-xs font-mono text-slate-500 dark:text-slate-400">scanner_gui.tsx</div>
                </div>
                
                <div className="relative aspect-[4/3] bg-gray-100 dark:bg-slate-950 overflow-hidden">
                  <img 
                    src="/plastic_bottle_waste.png" 
                    alt="Scanning plastic waste" 
                    className="w-full h-full object-cover opacity-90 dark:opacity-80"
                    loading="lazy"
                  />
                  {/* Bounding Box Mockup */}
                  <div className="absolute top-[20%] left-[25%] right-[30%] bottom-[15%] border-2 border-[#10B981] bg-[#10B981]/10 shadow-[0_0_15px_rgba(16,185,129,0.5)] flex flex-col justify-end">
                    <div className="absolute -top-6 -left-0.5 bg-[#10B981] text-white dark:text-[#121824] text-[10px] sm:text-[11px] font-bold px-2 py-1 font-mono tracking-wider flex items-center gap-1 shadow-sm">
                      <ScanLine className="w-3 h-3" />
                      PLASTIC
                      <span className="font-extrabold ml-1">98%</span>
                    </div>
                  </div>
                  
                  {/* Grid Lines Overlay */}
                  <div className="absolute inset-0 bg-[linear-gradient(rgba(16,185,129,0.05)_1px,transparent_1px),linear-gradient(90deg,rgba(16,185,129,0.05)_1px,transparent_1px)] bg-[length:20px_20px]"></div>
                </div>
                
                <div className="p-3 sm:p-4 bg-gray-50/90 dark:bg-slate-900/80 backdrop-blur-md flex items-center justify-between border-t border-gray-200 dark:border-slate-700/50">
                  <div className="flex items-center gap-2 sm:gap-3">
                    <div className="w-2 h-2 rounded-full bg-[#10B981] animate-pulse"></div>
                    <span className="text-xs sm:text-sm font-medium text-[#059669] dark:text-[#10B981]">Live Inference</span>
                  </div>
                  <div className="text-[10px] sm:text-xs text-slate-500 dark:text-slate-500 font-mono">24 FPS | Latency: 42ms</div>
                </div>
              </div>
              
              {/* Decorative Glow */}
              <div className="absolute -inset-10 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#10B981]/10 dark:from-[#10B981]/20 to-transparent blur-3xl -z-10 rounded-full"></div>
            </div>
          </div>
        </div>
      </section>

      <Suspense fallback={
        <div className="py-32 flex items-center justify-center bg-gray-50 dark:bg-[#121824] transition-colors">
          <div className="w-8 h-8 rounded-full border-2 border-[#10B981] border-t-transparent animate-spin"></div>
        </div>
      }>
        <ImpactSection />
        <CategoriesSection />
        <HowItWorksSection />
      </Suspense>
    </div>
  );
}
