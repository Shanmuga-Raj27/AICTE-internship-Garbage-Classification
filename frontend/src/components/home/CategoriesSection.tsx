import React, { useEffect, useState } from 'react';
import { Box, GlassWater, Wrench, Newspaper, ShoppingBag, Trash2, ChevronDown } from 'lucide-react';

const LOCAL_CATEGORIES = [
  { name: "Cardboard", description: "Boxes, packaging", icon: "Box", recyclable: true, tips: "Flatten before recycling." },
  { name: "Glass", description: "Bottles, jars", icon: "GlassWater", recyclable: true, tips: "Rinse clean." },
  { name: "Metal", description: "Cans, foil", icon: "Wrench", recyclable: true, tips: "Rinse food out." },
  { name: "Paper", description: "Newspapers", icon: "Newspaper", recyclable: true, tips: "Keep dry." },
  { name: "Plastic", description: "Bottles, bags", icon: "ShoppingBag", recyclable: true, tips: "Check recycle number." },
  { name: "Trash", description: "Non-recyclable", icon: "Trash2", recyclable: false, tips: "Goes to landfill." }
];

const ICON_MAP: Record<string, React.ElementType> = {
  // Primary name mapping (case-sensitive matching for local & API names)
  "Cardboard": Box,
  "Glass": GlassWater,
  "Metal": Wrench,
  "Paper": Newspaper,
  "Plastic": ShoppingBag,
  "Trash": Trash2,

  // Unicode Emoji mapping returned from Hugging Face Space API
  "📦": Box,
  "🍶": GlassWater,
  "🥫": Wrench,
  "📄": Newspaper,
  "🧴": ShoppingBag,
  "🗑️": Trash2,

  // Fallback compatibility string mapping
  "Box": Box,
  "Package": Box,
  "GlassWater": GlassWater,
  "Wine": GlassWater,
  "Wrench": Wrench,
  "Cog": Wrench,
  "Newspaper": Newspaper,
  "ShoppingBag": ShoppingBag,
  "Trash2": Trash2
};

export default function CategoriesSection() {
  const [categories, setCategories] = useState(LOCAL_CATEGORIES);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<number | null>(null);

  useEffect(() => {
    const fetchCategories = async () => {
      setLoading(true);
      try {
        const url = import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}/categories` : '';
        if (url) {
          const res = await fetch(url);
          if (res.ok) {
            const data = await res.json();
            setCategories(data.categories || data);
            return;
          }
        }
      } catch (err) {
        console.warn("API not accessible, using local categories fallback", err);
      } finally {
        setTimeout(() => setLoading(false), 800);
      }
    };
    fetchCategories();
  }, []);

  return (
    <section id="categories" className="bg-white dark:bg-[#121824] py-24 relative overflow-hidden border-t border-gray-200 dark:border-slate-800/50 transition-colors duration-300">
      <div className="absolute top-0 right-0 w-full h-full bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#10B981]/5 via-transparent dark:via-[#121824] to-transparent dark:to-[#121824] pointer-events-none"></div>
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center max-w-3xl mx-auto mb-16">
          <h2 className="text-3xl md:text-4xl font-bold text-[#0F172A] dark:text-[#F9FAFB] tracking-tight mb-4 font-display transition-colors duration-300">
            Supported Categories & <span className="text-[#10B981]">Disposal Guide</span>
          </h2>
          <p className="text-lg text-slate-700 dark:text-slate-300 font-light transition-colors duration-300">
            Explore the materials our AI deep learning model can detect and learn how to sort them optimally.
          </p>
        </div>

        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6 lg:gap-8">
          {loading ? (
            Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="rounded-2xl border border-gray-200 dark:border-slate-700/50 bg-gray-50 dark:bg-slate-800/30 flex flex-col p-6 h-[200px] animate-pulse">
                <div className="flex items-start justify-between mb-4">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-xl bg-gray-200 dark:bg-slate-700/50"></div>
                    <div className="w-24 h-6 bg-gray-200 dark:bg-slate-700/50 rounded-md"></div>
                  </div>
                  <div className="w-20 h-6 bg-gray-200 dark:bg-slate-700/50 rounded-full"></div>
                </div>
                <div className="space-y-3 mt-4">
                  <div className="w-full h-4 bg-gray-200 dark:bg-slate-700/50 rounded-md"></div>
                  <div className="w-2/3 h-4 bg-gray-200 dark:bg-slate-700/50 rounded-md"></div>
                </div>
              </div>
            ))
          ) : (
            categories.map((cat, idx) => {
              const Icon = ICON_MAP[cat.name] || ICON_MAP[cat.icon] || Box;
              const isExpanded = expandedId === idx;
              return (
                <div 
                  key={idx} 
                  onClick={() => setExpandedId(isExpanded ? null : idx)}
                  className={`group relative rounded-2xl border border-gray-200 dark:border-slate-700/50 bg-white dark:bg-slate-800/40 backdrop-blur-md shadow-sm hover:shadow-md dark:shadow-lg dark:hover:shadow-[0_0_30px_rgba(16,185,129,0.15)] hover:border-[#10B981]/40 dark:hover:border-[#10B981]/50 sm:hover:-translate-y-1.5 transition-all duration-300 overflow-hidden flex flex-col p-4 sm:p-6 cursor-pointer sm:cursor-default ${isExpanded ? 'h-auto' : 'h-[88px] sm:h-[200px]'}`}
                >
                  <div className="flex items-center justify-between z-10 relative sm:mb-4">
                    <div className="flex items-center gap-4">
                      <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-xl bg-gray-50 dark:bg-slate-900/80 border border-gray-100 dark:border-slate-700/80 flex items-center justify-center text-slate-700 dark:text-[#F9FAFB] group-hover:text-[#059669] dark:group-hover:text-[#10B981] group-hover:border-[#10B981]/30 transition-colors duration-300 shrink-0">
                        <Icon className="w-5 h-5 sm:w-6 sm:h-6" strokeWidth={1.5} />
                      </div>
                      <div className="flex flex-col sm:flex-row sm:items-center gap-1 sm:gap-3">
                        <h3 className="text-base sm:text-xl font-bold text-[#0F172A] dark:text-[#F9FAFB] transition-colors duration-300 line-clamp-1">{cat.name}</h3>
                        <span className={`px-2 py-0.5 sm:px-3 sm:py-1 text-[10px] sm:text-xs font-semibold rounded-full border self-start sm:self-auto ${
                          cat.recyclable 
                            ? 'bg-[#ECFDF5] text-[#059669] border-[#10B981]/30 dark:bg-amber-500/10 dark:text-amber-500 dark:border-amber-500/30' 
                            : 'bg-red-50 text-red-700 border-red-200 dark:bg-red-500/10 dark:text-red-500 dark:border-red-500/30'
                        }`}>
                          {cat.recyclable ? 'Recyclable' : 'Trash'}
                        </span>
                      </div>
                    </div>
                    <ChevronDown className={`w-5 h-5 text-slate-400 sm:hidden transition-transform duration-300 shrink-0 ${isExpanded ? 'rotate-180' : ''}`} />
                  </div>

                  <div className={`relative z-10 flex-1 sm:block transition-all duration-300 overflow-hidden ${isExpanded ? 'mt-4 opacity-100 max-h-40' : 'opacity-0 max-h-0 sm:opacity-100 sm:max-h-40 sm:mt-0'}`}>
                    <p className="text-sm text-slate-600 dark:text-slate-400 font-light mt-2 sm:group-hover:opacity-0 transition-opacity duration-300">
                      {cat.description}
                    </p>
                    <p className="text-sm text-[#0F172A] dark:text-[#F9FAFB] font-medium leading-relaxed transition-colors duration-300 mt-3 sm:hidden pb-1">
                      <span className="text-[#059669] dark:text-[#10B981] font-semibold">Action Required:</span> {cat.tips}
                    </p>
                  </div>

                  <div className="hidden sm:block absolute left-0 bottom-0 w-full p-6 bg-gradient-to-t from-gray-50 via-gray-50/95 to-gray-50/90 dark:from-slate-900 dark:via-slate-900/95 dark:to-slate-900/90 border-t border-[#10B981]/20 transform translate-y-[101%] group-hover:translate-y-0 transition-transform duration-300 ease-out z-20">
                    <p className="text-sm text-[#0F172A] dark:text-[#F9FAFB] font-medium leading-relaxed transition-colors duration-300">
                      <span className="text-[#059669] dark:text-[#10B981] font-semibold">Action Required:</span> {cat.tips}
                    </p>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </section>
  );
}
