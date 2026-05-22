import { ArrowRight, Github, Linkedin, Globe, Cpu, Layers, Box, Rocket } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer className="mt-auto border-t border-gray-200 dark:border-slate-800 bg-gray-50 dark:bg-[#0F141E] text-sm text-slate-700 dark:text-slate-400 font-sans transition-colors duration-300">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-12 gap-12 lg:gap-8 mb-16">
          
          {/* Column 1: Brand & Mission */}
          <div className="lg:col-span-4 flex flex-col items-start pr-0 lg:pr-8">
            <Link to="/" className="text-xl font-bold text-[#0F172A] dark:text-[#F9FAFB] tracking-tight mb-4 font-display transition-colors duration-300">
              Garbage Classification Platform
            </Link>
            <p className="text-slate-700 dark:text-slate-300 leading-relaxed font-light mb-6 transition-colors duration-300">
              Towards a Green India. Empowering citizens and industries to sort waste at the source using advanced Deep Learning.
            </p>
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-700/50 transition-colors duration-300">
              <div className="w-2 h-2 rounded-full bg-[#10B981] animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.8)]"></div>
              <span className="text-xs font-medium text-[#0F172A] dark:text-[#F9FAFB] tracking-wide transition-colors duration-300">FastAPI Backend: Online</span>
            </div>
          </div>

          {/* Column 2: Platform Navigation */}
          <div className="lg:col-span-2">
            <h4 className="text-xs font-semibold text-slate-700 dark:text-slate-400 uppercase tracking-widest mb-6 border-b border-gray-200 dark:border-slate-800 pb-2 transition-colors duration-300">Platform</h4>
            <ul className="space-y-4">
              {[
                { name: 'AI Classifier', href: '/classify' },
                { name: 'Supported Materials', href: '/#categories' },
                { name: 'Impact Data', href: '#' },
                { name: 'API Documentation', href: '#' },
              ].map((link, idx) => (
                <li key={idx}>
                  <Link 
                    to={link.href} 
                    className="group inline-flex items-center text-slate-700 dark:text-slate-300 hover:text-[#059669] dark:hover:text-[#10B981] transition-all duration-300"
                  >
                    <span className="relative overflow-hidden group-hover:translate-x-2 transition-transform duration-300">
                      {link.name}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Column 3: Engineering */}
          <div className="lg:col-span-3">
            <h4 className="text-xs font-semibold text-slate-700 dark:text-slate-400 uppercase tracking-widest mb-6 border-b border-gray-200 dark:border-slate-800 pb-2 transition-colors duration-300">Engineering</h4>
            <ul className="space-y-4">
              {[
                { name: 'GitHub Repository', icon: Github },
                { name: 'Hugging Face Space', icon: Rocket },
                { name: 'Netlify Deployment', icon: Globe },
                { name: 'Model Architecture', icon: Cpu },
              ].map((link, idx) => {
                const Icon = link.icon;
                return (
                  <li key={idx}>
                    <a 
                      href="#" 
                      className="group inline-flex items-center gap-2 text-slate-700 dark:text-slate-300 hover:text-[#059669] dark:hover:text-[#10B981] transition-all duration-300"
                    >
                      <Icon className="w-4 h-4 opacity-70 group-hover:opacity-100 transition-opacity" />
                      <span className="group-hover:translate-x-1 transition-transform duration-300">{link.name}</span>
                    </a>
                  </li>
                );
              })}
            </ul>
          </div>

          {/* Column 4: Collaboration / Connect */}
          <div className="lg:col-span-3">
            <h4 className="text-xs font-semibold text-slate-700 dark:text-slate-400 uppercase tracking-widest mb-6 border-b border-gray-200 dark:border-slate-800 pb-2 transition-colors duration-300">Connect</h4>
            <form className="mb-6 relative flex items-center group">
              <input 
                type="email" 
                placeholder="Enter email for eco-updates" 
                className="w-full bg-white dark:bg-slate-800/40 border border-gray-200 dark:border-slate-700/50 rounded-lg py-2.5 px-4 text-sm text-[#0F172A] dark:text-[#F9FAFB] placeholder-gray-400 dark:placeholder-slate-500 focus:outline-none focus:border-[#10B981]/50 focus:ring-1 focus:ring-[#10B981]/50 transition-all pr-12"
              />
              <button 
                type="submit" 
                className="absolute right-2 p-1.5 text-gray-500 dark:text-slate-400 hover:text-[#059669] dark:hover:text-[#10B981] hover:bg-gray-100 dark:hover:bg-[#10B981]/10 rounded-md transition-colors"
                aria-label="Subscribe"
              >
                <ArrowRight className="w-4 h-4" />
              </button>
            </form>
            <div className="flex items-center gap-4">
              <a href="#" className="w-9 h-9 rounded-full bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-700/50 flex items-center justify-center text-slate-600 dark:text-slate-300 hover:text-[#059669] dark:hover:text-[#F9FAFB] hover:border-[#10B981]/50 transition-all hover:-translate-y-1">
                <Github className="w-4 h-4" />
              </a>
              <a href="#" className="w-9 h-9 rounded-full bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-700/50 flex items-center justify-center text-slate-600 dark:text-slate-300 hover:text-[#059669] dark:hover:text-[#F9FAFB] hover:border-[#10B981]/50 transition-all hover:-translate-y-1">
                <Linkedin className="w-4 h-4" />
              </a>
              <a href="#" className="w-9 h-9 rounded-full bg-white dark:bg-slate-800/50 border border-gray-200 dark:border-slate-700/50 flex items-center justify-center text-slate-600 dark:text-slate-300 hover:text-[#059669] dark:hover:text-[#F9FAFB] hover:border-[#10B981]/50 transition-all hover:-translate-y-1">
                <Globe className="w-4 h-4" />
              </a>
            </div>
          </div>
          
        </div>

        {/* Bottom Copyright Bar */}
        <div className="pt-8 border-t border-gray-200 dark:border-slate-800 flex flex-col md:flex-row justify-between items-center gap-4 text-xs text-slate-600 dark:text-slate-400 transition-colors duration-300">
          <p className="text-center md:text-left transition-colors duration-300">
            &copy; 2026 Garbage Classification Platform. Built for the Swachh Bharat Mission initiative.
          </p>
          <div className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-white dark:bg-slate-800/30 border border-gray-200 dark:border-slate-700/30 transition-colors duration-300">
            <Layers className="w-3.5 h-3.5 text-[#10B981]" />
            <span className="font-medium text-slate-800 dark:text-slate-300 transition-colors duration-300">Designed with React & Tailwind CSS</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
