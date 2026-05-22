import React from 'react';
import { Camera, Cpu, Recycle } from 'lucide-react';

export default function HowItWorksSection() {
  return (
    <section id="how-it-works" className="bg-gray-50 dark:bg-[#121824] py-24 relative overflow-hidden border-t border-gray-200 dark:border-slate-800/50 transition-colors duration-300">
      <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-[#10B981]/5 via-transparent dark:via-[#121824] to-transparent dark:to-[#121824] pointer-events-none"></div>
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-20">
          <h2 className="text-3xl md:text-4xl font-bold text-[#0F172A] dark:text-[#F9FAFB] tracking-tight mb-4 font-display transition-colors duration-300">
            The AI Sorting Pipeline
          </h2>
          <p className="text-lg text-slate-700 dark:text-slate-300 font-light transition-colors duration-300">
            Three simple steps to process, analyze, and intelligently classify your waste in real-time.
          </p>
        </div>

        <div className="relative">
          {/* Horizontal Line for Desktop / Vertical for Mobile */}
          <div className="hidden md:block absolute top-[48px] left-[16.66%] right-[16.66%] border-t-2 border-dashed border-gray-300 dark:border-[#10B981]/30 transition-colors duration-300"></div>
          <div className="md:hidden absolute top-10 bottom-10 left-[39px] w-0.5 border-l-2 border-dashed border-gray-300 dark:border-[#10B981]/30 transition-colors duration-300"></div>
          
          <div className="grid md:grid-cols-3 gap-8 md:gap-12 relative z-10">
            {/* Step 1 */}
            <div className="flex flex-row md:flex-col items-center md:items-center md:text-center group gap-6 md:gap-0">
              <div className="w-20 h-20 md:w-24 md:h-24 shrink-0 rounded-2xl bg-white dark:bg-slate-800/80 border border-gray-200 dark:border-slate-700/80 shadow-sm flex items-center justify-center md:mb-6 relative hover:border-[#10B981]/40 dark:hover:border-[#10B981]/50 hover:shadow-md dark:hover:shadow-[0_0_30px_rgba(16,185,129,0.2)] transition-all duration-300">
                <div className="absolute -top-2 -right-2 md:-top-3 md:-right-3 w-6 h-6 md:w-8 md:h-8 rounded-full bg-[#10B981] dark:bg-[#A3E635] text-white dark:text-[#121824] text-sm md:text-base font-bold flex items-center justify-center font-mono border-2 md:border-4 border-gray-50 dark:border-[#121824]">1</div>
                <Camera className="w-8 h-8 md:w-10 md:h-10 text-slate-600 dark:text-[#F9FAFB] group-hover:text-[#059669] dark:group-hover:text-[#10B981] transition-colors duration-300" strokeWidth={1.5} />
              </div>
              <div className="flex flex-col text-left md:text-center">
                <h3 className="text-lg md:text-xl font-bold text-[#0F172A] dark:text-[#F9FAFB] mb-2 md:mb-3 transition-colors duration-300">Snap or Upload</h3>
                <p className="text-sm md:text-base text-slate-700 dark:text-slate-300 leading-relaxed font-light transition-colors duration-300">
                  Take a photo of the waste item directly through your device camera or drop an image into the responsive React frontend.
                </p>
              </div>
            </div>

            {/* Step 2 */}
            <div className="flex flex-row md:flex-col items-center md:items-center md:text-center group gap-6 md:gap-0">
              <div className="w-20 h-20 md:w-24 md:h-24 shrink-0 rounded-2xl bg-white dark:bg-slate-800/80 border border-gray-200 dark:border-slate-700/80 shadow-sm flex items-center justify-center md:mb-6 relative hover:border-[#10B981]/40 dark:hover:border-[#10B981]/50 hover:shadow-md dark:hover:shadow-[0_0_30px_rgba(16,185,129,0.2)] transition-all duration-300 z-10">
                <div className="absolute -top-2 -right-2 md:-top-3 md:-right-3 w-6 h-6 md:w-8 md:h-8 rounded-full bg-[#10B981] dark:bg-[#A3E635] text-white dark:text-[#121824] text-sm md:text-base font-bold flex items-center justify-center font-mono border-2 md:border-4 border-gray-50 dark:border-[#121824]">2</div>
                <Cpu className="w-8 h-8 md:w-10 md:h-10 text-slate-600 dark:text-[#F9FAFB] group-hover:text-[#059669] dark:group-hover:text-[#10B981] transition-colors duration-300" strokeWidth={1.5} />
              </div>
              <div className="flex flex-col text-left md:text-center">
                <h3 className="text-lg md:text-xl font-bold text-[#0F172A] dark:text-[#F9FAFB] mb-2 md:mb-3 transition-colors duration-300">Deep Learning Analysis</h3>
                <p className="text-sm md:text-base text-slate-700 dark:text-slate-300 leading-relaxed font-light transition-colors duration-300">
                  The image is securely routed to our backend, where an optimized EfficientNetV2B2 model scans the visual features with over 90% accuracy.
                </p>
              </div>
            </div>

            {/* Step 3 */}
            <div className="flex flex-row md:flex-col items-center md:items-center md:text-center group gap-6 md:gap-0">
              <div className="w-20 h-20 md:w-24 md:h-24 shrink-0 rounded-2xl bg-white dark:bg-slate-800/80 border border-gray-200 dark:border-slate-700/80 shadow-sm flex items-center justify-center md:mb-6 relative hover:border-[#10B981]/40 dark:hover:border-[#10B981]/50 hover:shadow-md dark:hover:shadow-[0_0_30px_rgba(16,185,129,0.2)] transition-all duration-300 z-10">
                <div className="absolute -top-2 -right-2 md:-top-3 md:-right-3 w-6 h-6 md:w-8 md:h-8 rounded-full bg-[#10B981] dark:bg-[#A3E635] text-white dark:text-[#121824] text-sm md:text-base font-bold flex items-center justify-center font-mono border-2 md:border-4 border-gray-50 dark:border-[#121824]">3</div>
                <Recycle className="w-8 h-8 md:w-10 md:h-10 text-slate-600 dark:text-[#F9FAFB] group-hover:text-[#059669] dark:group-hover:text-[#10B981] transition-colors duration-300" strokeWidth={1.5} />
              </div>
              <div className="flex flex-col text-left md:text-center">
                <h3 className="text-lg md:text-xl font-bold text-[#0F172A] dark:text-[#F9FAFB] mb-2 md:mb-3 transition-colors duration-300">Smart Sorting Results</h3>
                <p className="text-sm md:text-base text-slate-700 dark:text-slate-300 leading-relaxed font-light transition-colors duration-300">
                  Receive an instant classification response alongside dynamic, actionable disposal recommendations to ensure it reaches the proper circular economy stream.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
