import { Menu, X, Leaf, Cpu, ArrowLeft, Sun, Moon } from 'lucide-react';
import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTheme } from './ThemeProvider';

export default function Navbar() {
  const location = useLocation();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const { theme, setTheme } = useTheme();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const isActive = (href: string) => {
    if (href === '/') return location.pathname === '/' && !location.hash;
    if (href.startsWith('/#')) return location.pathname === '/' && location.hash === href.replace('/', '');
    return location.pathname === href;
  };

  const navLinks = [
    { name: 'Home', href: '/' },
    { name: 'The Impact', href: '/#impact' },
    { name: 'Supported Categories', href: '/#categories' },
    { name: 'How It Works', href: '/#how-it-works' },
  ];

  const handleNavLinkClick = (href: string) => {
    setMobileMenuOpen(false);
    if (href === '/' && location.pathname === '/') {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } else if (href.startsWith('/#') && location.pathname === '/') {
      const id = href.replace('/#', '');
      const element = document.getElementById(id);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  };

  if (location.pathname === '/classify') {
    return (
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white dark:bg-[#121824] border-b border-gray-200 dark:border-[#10B981]/30 h-14 flex items-center justify-center shadow-sm dark:shadow-lg transition-colors duration-300">
        <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex justify-between items-center">
          {/* Navigation Control */}
          <Link
            to="/"
            className="flex items-center min-w-[48px] min-h-[48px] md:min-h-0 md:min-w-0 gap-2 text-sm font-semibold text-[#0F172A] dark:text-[#F9FAFB] group px-3 py-1.5 rounded-lg border border-gray-200 dark:border-[#10B981]/20 bg-gray-50 dark:bg-[#10B981]/5 hover:bg-white dark:hover:bg-[#10B981] hover:border-[#10B981] hover:text-[#10B981] dark:hover:text-[#121824] hover:shadow-sm dark:hover:shadow-[0_0_15px_rgba(16,185,129,0.4)] transition-all duration-300"
          >
            <ArrowLeft className="w-4 h-4 group-hover:-translate-x-0.5 transition-transform duration-300" />
            <span className="hidden sm:inline">Back to Home</span>
          </Link>

          {/* Center Metadata Display */}
          <div className="hidden md:flex items-center gap-2 text-xs font-mono text-slate-700 dark:text-slate-300 bg-gray-50 dark:bg-slate-800/40 px-4 py-1.5 rounded-md border border-gray-200 dark:border-slate-700/50 transition-colors duration-300">
            <Cpu className="w-3.5 h-3.5 text-[#10B981] dark:text-[#A3E635]" />
            AI Environment: EfficientNetV2B2 Inference Engine
          </div>

          {/* Connection Status & Theme */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-xs font-semibold px-3 py-1.5 rounded-full bg-[#ECFDF5] dark:bg-[#10B981]/10 border border-[#10B981]/30">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#10B981] opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-[#10B981]"></span>
              </span>
              <span className="text-[#10B981] hidden sm:inline">FastAPI Backend: Connected</span>
              <span className="text-[#10B981] sm:hidden">Connected</span>
            </div>
            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="p-2 rounded-full text-slate-600 hover:bg-gray-100 dark:text-[#F9FAFB] dark:hover:bg-slate-800 transition-colors duration-300 min-w-[48px] min-h-[48px] md:min-w-0 md:min-h-0 flex items-center justify-center"
              aria-label="Toggle Theme"
            >
              {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </nav>
    );
  }

  return (
    <nav className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 border-b ${
      isScrolled 
        ? 'bg-white/90 dark:bg-[#121824]/90 backdrop-blur-md border-gray-200 dark:border-[#10B981]/30 shadow-sm dark:shadow-lg' 
        : 'bg-white/50 dark:bg-[#121824]/50 backdrop-blur-sm border-gray-100 dark:border-slate-800/50'
    }`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-20 items-center">
          
          {/* Brand */}
          <Link to="/" className="flex items-center gap-3 group px-2 py-1 rounded-xl focus:outline-none focus:ring-2 focus:ring-[#10B981]">
            <div className="relative w-10 h-10 flex items-center justify-center bg-gray-100 dark:bg-slate-800/50 rounded-xl border border-gray-200 dark:border-slate-700/50 group-hover:border-[#10B981]/80 transition-all duration-300 group-hover:shadow-[0_0_15px_rgba(16,185,129,0.3)]">
              <Cpu className="w-5 h-5 text-[#0F172A] dark:text-[#F9FAFB] group-hover:text-[#10B981] transition-colors duration-300" strokeWidth={1.5} />
              <Leaf className="w-3.5 h-3.5 text-[#10B981] dark:text-[#A3E635] absolute bottom-1 right-1" strokeWidth={3} />
            </div>
            <span className="font-bold text-lg md:text-xl text-[#0F172A] dark:text-[#F9FAFB] tracking-tight font-display group-hover:text-[#10B981] dark:group-hover:text-[#F9FAFB] transition-colors">
              Garbage Classifier
            </span>
          </Link>

          {/* Desktop Nav */}
          <div className="hidden lg:flex items-center gap-8">
            {navLinks.map((link) => {
              const active = isActive(link.href);
              return (
                <Link
                  key={link.name}
                  to={link.href}
                  onClick={() => handleNavLinkClick(link.href)}
                  className={`relative text-sm font-semibold tracking-wide transition-colors duration-300 ${active ? 'text-[#059669] dark:text-[#A3E635]' : 'text-slate-700 dark:text-[#F9FAFB] hover:text-[#059669] dark:hover:text-[#A3E635]'}`}
                >
                  {link.name}
                  {active && (
                    <span className="absolute -bottom-1.5 left-1/2 -translate-x-1/2 w-1.5 h-1.5 rounded-full bg-[#10B981] shadow-[0_0_8px_rgba(16,185,129,0.8)]"></span>
                  )}
                </Link>
              );
            })}
          </div>

          {/* CTA & Mobile Toggle */}
          <div className="flex items-center gap-2 sm:gap-4">
            <button
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
              className="hidden lg:flex p-2 rounded-full text-slate-600 hover:bg-gray-100 dark:text-[#F9FAFB] dark:hover:bg-slate-800 transition-colors duration-300 min-w-[48px] min-h-[48px] items-center justify-center"
              aria-label="Toggle Theme"
            >
              {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            <Link 
              to="/classify" 
              className="hidden sm:flex items-center justify-center bg-[#10B981] dark:bg-[#F9FAFB] text-white dark:hover:bg-[#A3E635] dark:text-[#121824] px-6 py-2.5 rounded-full font-bold text-sm transition-all duration-300 hover:bg-[#059669] hover:shadow-md dark:hover:shadow-[0_0_20px_-5px_rgba(163,230,53,0.6)] hover:-translate-y-0.5 min-h-[48px]"
            >
              Launch AI Classifier
            </Link>
            
            <button 
              className="lg:hidden p-2 text-[#10B981] hover:text-[#059669] dark:text-[#10B981] dark:hover:text-[#059669] transition-colors min-w-[48px] min-h-[48px] flex items-center justify-center bg-gray-50 dark:bg-slate-800/40 rounded-xl border border-gray-200 dark:border-slate-700/50"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              aria-label="Toggle Menu"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu Drawer */}
      <div 
        className={`lg:hidden absolute top-20 left-0 w-full bg-white/95 dark:bg-[#121824]/95 backdrop-blur-xl border-b border-gray-200 dark:border-[#10B981]/30 transition-all duration-300 overflow-hidden ${
          mobileMenuOpen ? 'max-h-[500px] opacity-100 py-4' : 'max-h-0 opacity-0 py-0'
        }`}
      >
        <div className="px-4 flex flex-col gap-2">
          {navLinks.map((link) => {
             const active = isActive(link.href);
             return (
              <Link
                key={link.name}
                to={link.href}
                onClick={() => handleNavLinkClick(link.href)}
                className={`text-base font-semibold px-4 py-4 rounded-xl transition-colors min-h-[48px] flex items-center ${
                  active ? 'bg-[#ECFDF5] dark:bg-[#10B981]/15 text-[#059669] dark:text-[#A3E635] border border-[#10B981]/30' : 'text-slate-700 dark:text-[#F9FAFB] hover:bg-gray-50 dark:hover:bg-slate-800/80 hover:text-[#059669] dark:hover:text-[#A3E635] border border-transparent'
                }`}
              >
                {link.name}
              </Link>
             );
          })}
          
          <div className="flex items-center justify-between px-4 py-4 mb-2 min-h-[48px]">
            <span className="text-slate-700 dark:text-[#F9FAFB] font-semibold">Theme</span>
            <button
              onClick={() => {
                setTheme(theme === 'dark' ? 'light' : 'dark');
              }}
              className="p-2 rounded-full bg-gray-100 dark:bg-slate-800 text-[#0F172A] dark:text-[#F9FAFB] transition-colors min-w-[48px] min-h-[48px] flex items-center justify-center"
            >
              {theme === 'dark' ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>

          <div className="pt-4 pb-2 px-1 mt-2 border-t border-gray-200 dark:border-slate-800">
            <Link 
              to="/classify"
              onClick={() => setMobileMenuOpen(false)}
              className="flex justify-center items-center w-full bg-[#10B981] text-white dark:bg-[#A3E635] dark:text-[#121824] px-4 py-4 rounded-xl font-bold text-base shadow-sm dark:shadow-[0_0_20px_-5px_rgba(163,230,53,0.5)] active:scale-[0.98] transition-all min-h-[48px]"
            >
              Launch AI Classifier
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
