import React from 'react';
import { Trash, Recycle, BrainCircuit } from 'lucide-react';

export default function ImpactSection() {
  return (
    <section id="impact" className="bg-gray-50 dark:bg-[#121824] py-20 relative border-t border-gray-200 dark:border-slate-800/50 transition-colors duration-300">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_left,_var(--tw-gradient-stops))] from-[#10B981]/5 via-transparent dark:via-[#121824] to-transparent dark:to-[#121824]"></div>
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-[#0F172A] dark:text-[#F9FAFB] tracking-tight font-display mb-4 transition-colors duration-300">
            Making an Impact: <span className="text-[#10B981]">The Challenge</span> and <span className="text-[#10B981] dark:text-[#A3E635]">The Solution</span>
          </h2>
          <p className="text-lg text-slate-700 dark:text-slate-300 max-w-2xl mx-auto font-light transition-colors duration-300">
            We stand at a critical intersection of ecological necessity and technological capability. Here is the reality we face and how AI provides a path forward.
          </p>
        </div>

        <div className="flex overflow-x-auto md:grid md:grid-cols-3 gap-4 md:gap-6 lg:gap-8 mt-4 snap-x snap-mandatory pb-6 md:pb-0 -mx-4 px-4 md:mx-0 md:px-0 scrollbar-hide">
          {/* Card 1 */}
          <div className="min-w-[85vw] sm:min-w-[320px] md:min-w-0 snap-center bg-white dark:bg-slate-800/40 backdrop-blur-md rounded-2xl p-8 border border-gray-100 dark:border-slate-700/50 hover:border-[#10B981]/30 dark:hover:border-[#10B981]/50 transition-all shadow-sm hover:shadow-md dark:shadow-lg relative overflow-hidden group">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-gray-200 dark:via-slate-600 to-transparent group-hover:via-[#10B981] transition-colors"></div>
            <div className="w-14 h-14 bg-gray-50 dark:bg-slate-900/80 rounded-xl flex items-center justify-center mb-6 border border-gray-100 dark:border-slate-700 transition-colors duration-300">
              <Trash className="w-7 h-7 text-[#059669] dark:text-[#10B981]" strokeWidth={1.5} />
            </div>
            <h3 className="text-xl font-semibold text-[#0F172A] dark:text-[#F9FAFB] mb-2 transition-colors duration-300">Waste Generation</h3>
            <p className="text-4xl font-bold text-[#059669] dark:text-[#A3E635] mb-3 font-mono tracking-tight transition-colors duration-300">62M <span className="text-lg text-[#059669] dark:text-[#10B981] uppercase tracking-widest font-sans">Tonnes</span></p>
            <p className="text-slate-700 dark:text-slate-300 leading-relaxed transition-colors duration-300">India generates significant waste annually, putting immense strain on current landfill capacities.</p>
          </div>

          {/* Card 2 */}
          <div className="min-w-[85vw] sm:min-w-[320px] md:min-w-0 snap-center bg-white dark:bg-slate-800/40 backdrop-blur-md rounded-2xl p-8 border border-gray-100 dark:border-slate-700/50 hover:border-[#10B981]/30 dark:hover:border-[#10B981]/50 transition-all shadow-sm hover:shadow-md dark:shadow-lg relative overflow-hidden group">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-gray-200 dark:via-slate-600 to-transparent group-hover:via-[#A3E635] transition-colors"></div>
            <div className="w-14 h-14 bg-gray-50 dark:bg-slate-900/80 rounded-xl flex items-center justify-center mb-6 border border-gray-100 dark:border-slate-700 transition-colors duration-300">
              <Recycle className="w-7 h-7 text-[#059669] dark:text-[#A3E635]" strokeWidth={1.5} />
            </div>
            <h3 className="text-xl font-semibold text-[#0F172A] dark:text-[#F9FAFB] mb-2 transition-colors duration-300">Waste Treatment</h3>
            <p className="text-4xl font-bold text-[#059669] dark:text-[#A3E635] mb-3 font-mono tracking-tight transition-colors duration-300">20% <span className="text-lg text-[#059669] dark:text-[#10B981] uppercase tracking-widest font-sans">Treated</span></p>
            <p className="text-slate-700 dark:text-slate-300 leading-relaxed transition-colors duration-300">Only about 20% of waste is currently treated, leaving massive room for improved processing efficiency.</p>
          </div>

          {/* Card 3 */}
          <div className="min-w-[85vw] sm:min-w-[320px] md:min-w-0 snap-center bg-white dark:bg-slate-800/40 backdrop-blur-md rounded-2xl p-8 border border-gray-100 dark:border-slate-700/50 hover:border-[#10B981]/30 dark:hover:border-[#10B981]/50 transition-all shadow-sm hover:shadow-md dark:shadow-lg relative overflow-hidden group">
            <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-gray-200 dark:via-slate-600 to-transparent group-hover:via-[#10B981] transition-colors"></div>
            <div className="w-14 h-14 bg-gray-50 dark:bg-slate-900/80 rounded-xl flex items-center justify-center mb-6 border border-gray-100 dark:border-slate-700 transition-colors duration-300">
              <BrainCircuit className="w-7 h-7 text-[#059669] dark:text-[#10B981]" strokeWidth={1.5} />
            </div>
            <h3 className="text-xl font-semibold text-[#0F172A] dark:text-[#F9FAFB] mb-2 transition-colors duration-300">Classification Accuracy</h3>
            <p className="text-4xl font-bold text-[#059669] dark:text-[#10B981] mb-3 font-mono tracking-tight transition-colors duration-300">&gt;90% <span className="text-lg text-[#059669] dark:text-[#A3E635] uppercase tracking-widest font-sans">Precise</span></p>
            <p className="text-slate-700 dark:text-slate-300 leading-relaxed transition-colors duration-300">Leveraging Transfer Learning for precise sorting.</p>
          </div>
        </div>
      </div>
    </section>
  );
}
