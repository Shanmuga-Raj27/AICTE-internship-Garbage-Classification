import React, { useState, useRef, useCallback, DragEvent, useEffect } from 'react';
import { UploadCloud, Image as ImageIcon, Camera, RotateCcw, AlertCircle, Loader2, CheckCircle2, ChevronRight, Info, ScanLine, Package, Wine, Cog, Newspaper, ShoppingBag, Trash, Eye, Activity } from 'lucide-react';

const CATEGORY_MAP: Record<string, any> = {
  Cardboard: { icon: Package, recyclable: true, tips: "Please flatten boxes and ensure they are free from grease or food residue before placing in the dry waste bin.", binType: "Dry Recycling Bin" },
  Glass: { icon: Wine, recyclable: true, tips: "Rinse out any residual liquids. Remove metal or plastic caps and sort separately.", binType: "Glass Recycling Bin" },
  Metal: { icon: Cog, recyclable: true, tips: "Rinse food out completely. Crush aluminum cans to save space if possible.", binType: "Dry Recycling Bin" },
  Paper: { icon: Newspaper, recyclable: true, tips: "Keep dry and unsoiled. Shredded paper should be contained in a paper bag.", binType: "Dry Recycling Bin" },
  Plastic: { icon: ShoppingBag, recyclable: true, tips: "Please rinse out any residual liquids and deposit this item in your blue sorting bin to support the circular economy.", binType: "Dry Recycling Bin" },
  Trash: { icon: Trash, recyclable: false, tips: "Dispose of this in standard mixed waste bins. Ensure tightly sealed to prevent leakage.", binType: "General Waste" }
};

const SAMPLE_IMAGES = [
  { id: 1, name: "Plastic Bottle", url: "/plastic_bottle_waste.png" },
  { id: 2, name: "Newspaper", url: "https://images.unsplash.com/photo-1532153955177-f59af40d6472?auto=format&fit=crop&w=300&q=80" },
  { id: 3, name: "Glass Jar", url: "/glass_039.jpg" },
  { id: 4, name: "Cardboard Box", url: "/cardboard_195.jpg" }
];

type ClassificationStatus = 'idle' | 'loading' | 'success' | 'error';

export default function Classifier() {
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [status, setStatus] = useState<ClassificationStatus>('idle');
  const [result, setResult] = useState<any>(null);
  const [errorMessage, setErrorMessage] = useState('');
  const [isDragging, setIsDragging] = useState(false);

  // Live Camera states
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Clean up camera stream on unmount
  useEffect(() => {
    return () => {
      if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
      }
    };
  }, [cameraStream]);

  const startCamera = async () => {
    setIsCameraActive(true);
    setErrorMessage('');
    setImageSrc(null);
    setResult(null);
    setStatus('idle');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' }
      });
      setCameraStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err: any) {
      console.error("Camera access error:", err);
      setErrorMessage("Could not access camera. Please check permissions.");
      setIsCameraActive(false);
    }
  };

  const stopCamera = useCallback(() => {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      setCameraStream(null);
    }
    setIsCameraActive(false);
  }, [cameraStream]);

  const capturePhoto = () => {
    if (videoRef.current) {
      const video = videoRef.current;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        canvas.toBlob((blob) => {
          if (blob) {
            const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
            const dataUrl = canvas.toDataURL('image/jpeg');
            setImageSrc(dataUrl);
            stopCamera();
            runClassification(file);
            setTimeout(() => {
              document.getElementById('results-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
          }
        }, 'image/jpeg', 0.95);
      }
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setErrorMessage('Please upload a valid image file (JPG, PNG).');
      setStatus('error');
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setErrorMessage('File size exceeds the 5MB limit. Please upload a smaller image.');
      setStatus('error');
      return;
    }
    const reader = new FileReader();
    reader.onload = (event) => {
      setImageSrc(event.target?.result as string);
      setStatus('idle');
      setResult(null);
      runClassification(file); // Auto-start classification
      setTimeout(() => {
        document.getElementById('results-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const fetchImageAsFile = async (url: string, filename: string): Promise<File> => {
    const response = await fetch(url);
    const blob = await response.blob();
    return new File([blob], filename, { type: blob.type || 'image/jpeg' });
  };

  const selectSample = async (url: string, name: string) => {
    setImageSrc(url);
    setStatus('loading');
    setResult(null);
    setTimeout(() => {
      document.getElementById('results-panel')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);

    try {
      const file = await fetchImageAsFile(url, `${name.toLowerCase().replace(/\s+/g, '_')}.jpg`);
      await runClassification(file);
    } catch (err: any) {
      setErrorMessage(err.message || 'Failed to process sample image');
      setStatus('error');
    }
  };

  const clearWorkspace = () => {
    setImageSrc(null);
    setResult(null);
    setStatus('idle');
    setIsDragging(false);
  };

  const runClassification = async (file: File) => {
    setStatus('loading');
    setErrorMessage('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'https://shanmugaraj27-garbage-classification-backend.hf.space';
      const response = await fetch(`${apiUrl}/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || data.error || 'Prediction failed');
      }

      const categoryName = data.prediction; // e.g. "Cardboard"
      const mappedInfo = CATEGORY_MAP[categoryName] || {
        icon: Package,
        recyclable: data.category_info?.recyclable ?? false,
        tips: data.category_info?.tips ?? 'No instructions available.',
        binType: 'General Waste'
      };

      // confidence is float (e.g. 0.95), convert to percentage
      const confVal = data.confidence ? parseFloat((data.confidence * 100).toFixed(1)) : 0;

      setResult({
        category: categoryName,
        confidence: confVal,
        ...mappedInfo
      });
      setStatus('success');
    } catch (err: any) {
      setErrorMessage(err.message || 'An error occurred during prediction.');
      setStatus('error');
    }
  };

  return (
    <div className="flex-1 bg-gray-50 dark:bg-[#121824] pb-10 pt-20 min-h-[calc(100vh-56px)] overflow-x-hidden transition-colors duration-300">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">

        {/* Header */}
        <div className="mb-6 lg:mb-10 text-center lg:text-left flex flex-col justify-between items-center lg:items-start border-b border-gray-200 dark:border-slate-800 pb-6 transition-colors duration-300">
          <h1 className="text-2xl lg:text-3xl font-bold text-[#0F172A] dark:text-[#F9FAFB] tracking-tight font-display mb-3 transition-colors duration-300">AI Garbage Classifier</h1>
          <p className="text-slate-600 dark:text-slate-300 font-light flex flex-wrap items-center justify-center lg:justify-start gap-3 lg:gap-4 text-xs lg:text-sm transition-colors duration-300">
            <span className="inline-flex items-center gap-1.5"><CpuIcon className="w-3.5 h-3.5 lg:w-4 lg:h-4 text-[#059669] dark:text-[#A3E635]" /> Model: EfficientNetV2</span>
            <span className="inline-flex items-center gap-1.5"><Activity className="w-3.5 h-3.5 lg:w-4 lg:h-4 text-[#059669] dark:text-[#10B981]" /> Status: Operational</span>
            <span className="inline-flex items-center gap-1.5"><ScanLine className="w-3.5 h-3.5 lg:w-4 lg:h-4 text-slate-500 dark:text-slate-400" /> Latency: ~120ms</span>
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-6 lg:gap-8 items-stretch">

          {/* Left Column: The Intelligent Upload Workspace */}
          <div className="flex flex-col gap-4 lg:gap-6">
            <div className="bg-white dark:bg-[#1F2937] rounded-2xl shadow-xl dark:shadow-2xl border border-gray-200 dark:border-slate-700/50 p-4 lg:p-5 flex flex-col h-[360px] lg:h-[440px] w-full overflow-hidden transition-colors duration-300 relative">

              {isCameraActive ? (
                <div className="w-full h-full relative rounded-xl overflow-hidden bg-black flex-1 flex flex-col">
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    className="w-full h-full object-cover flex-1"
                  />

                  {/* Camera Controls Overlay */}
                  <div className="absolute bottom-3 left-0 right-0 flex justify-center items-center gap-6 px-4">
                    <button
                      type="button"
                      onClick={stopCamera}
                      className="bg-white/20 hover:bg-white/30 active:bg-white/40 text-white backdrop-blur-md px-3.5 py-1.5 rounded-lg text-xs font-semibold border border-white/20 transition-all"
                    >
                      Cancel
                    </button>

                    <button
                      type="button"
                      onClick={capturePhoto}
                      className="w-12 h-12 rounded-full bg-white hover:bg-gray-100 border-4 border-[#10B981] flex items-center justify-center shadow-lg active:scale-95 transition-all"
                      aria-label="Capture Photo"
                    >
                      <div className="w-7 h-7 rounded-full bg-[#10B981]"></div>
                    </button>

                    <div className="w-14"></div> {/* Spacer for symmetry */}
                  </div>
                </div>
              ) : !imageSrc ? (
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className={`flex-1 flex flex-col items-center justify-center border-2 border-dashed border-[#10B981] rounded-2xl p-4 lg:p-6 cursor-pointer transition-all duration-300 relative`}
                >
                  <div className="flex flex-col items-center justify-center flex-1">
                    <div className="w-10 h-10 lg:w-12 lg:h-12 rounded-full flex items-center justify-center mb-3 lg:mb-4 bg-gray-100 dark:bg-slate-800 transition-colors duration-300">
                      <UploadCloud className="w-5 h-5 lg:w-6 lg:h-6 text-slate-500 dark:text-slate-400" strokeWidth={1.5} />
                    </div>
                    <h3 className="text-base lg:text-lg font-bold text-[#0F172A] dark:text-[#F9FAFB] mb-1.5 transition-colors duration-300 text-center">Tap to upload or drag & drop</h3>
                    <p className="text-[11px] lg:text-xs text-slate-500 dark:text-slate-400 text-center transition-colors duration-300">Supports JPG, PNG up to 5MB.</p>

                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation(); // Prevent launching the file picker
                        startCamera();
                      }}
                      className="mt-4 flex items-center justify-center gap-2 bg-[#10B981] hover:bg-[#059669] text-white dark:text-[#121824] dark:bg-[#A3E635] dark:hover:bg-[#bbf7d0] px-4 py-2 rounded-xl text-xs font-bold transition-all shadow-sm hover:shadow"
                    >
                      <Camera className="w-4 h-4" />
                      Take Photo
                    </button>
                  </div>

                  <input
                    type="file"
                    accept="image/jpeg, image/png"
                    className="hidden"
                    ref={fileInputRef}
                    onChange={handleFileUpload}
                  />
                </div>
              ) : (
                <div className="w-full relative rounded-xl overflow-hidden border border-gray-200 dark:border-slate-700 bg-gray-100 dark:bg-slate-900 group transition-colors duration-300 flex-1">
                  <img src={imageSrc} alt="Uploaded waste" className="w-full h-full object-cover dark:opacity-80" />

                  {/* Scanner Animation during loading on the left side */}
                  {status === 'loading' && (
                    <>
                      <div className="absolute inset-0 bg-[#10B981]/10 dark:bg-[#A3E635]/10 animate-pulse"></div>
                      <div className="absolute left-0 right-0 h-1 bg-[#10B981] dark:bg-[#A3E635] shadow-[0_0_15px_rgba(16,185,129,0.8)] dark:shadow-[0_0_15px_rgba(163,230,53,0.8)] animate-[scan_2s_ease-in-out_infinite]"></div>
                      <div className="absolute inset-x-8 top-[25%] bottom-[25%] border-2 border-[#10B981]/30 dark:border-[#A3E635]/30">
                        <div className="absolute -top-1 -left-1 w-2 h-2 border-t-2 border-l-2 border-[#10B981] dark:border-[#A3E635]"></div>
                        <div className="absolute -top-1 -right-1 w-2 h-2 border-t-2 border-r-2 border-[#10B981] dark:border-[#A3E635]"></div>
                        <div className="absolute -bottom-1 -left-1 w-2 h-2 border-b-2 border-l-2 border-[#10B981] dark:border-[#A3E635]"></div>
                        <div className="absolute -bottom-1 -right-1 w-2 h-2 border-b-2 border-r-2 border-[#10B981] dark:border-[#A3E635]"></div>
                      </div>

                      <div className="absolute bottom-3 left-3 right-3 bg-white/95 dark:bg-slate-900/90 backdrop-blur-md rounded-xl p-2.5 sm:p-3 border border-gray-200 dark:border-slate-700/50 flex flex-col transition-colors duration-300">
                        <div className="flex items-center gap-2 mb-1.5">
                          <Loader2 className="w-3.5 h-3.5 animate-spin text-[#10B981] dark:text-[#A3E635]" />
                          <span className="text-xs text-[#0F172A] dark:text-[#F9FAFB] font-medium tracking-wide">Processing Tensor Maps...</span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-slate-800 rounded-full h-1 overflow-hidden transition-colors duration-300">
                          <div className="bg-[#10B981] dark:bg-[#A3E635] h-1 rounded-full animate-[progress_2s_ease-in-out_infinite]"></div>
                        </div>
                      </div>
                    </>
                  )}

                  {/* Reset/Change Actions */}
                  {status !== 'loading' && (
                    <div className="absolute top-3 right-3 flex gap-2">
                      <button
                        onClick={clearWorkspace}
                        className="bg-white/95 dark:bg-slate-900/95 hover:bg-red-50 dark:hover:bg-red-500/80 backdrop-blur text-slate-800 dark:text-white p-2 rounded-lg transition-colors border border-gray-200 dark:border-slate-700 hover:border-red-500 hover:text-red-600 shadow-sm"
                        title="Clear image"
                      >
                        <RotateCcw className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  )}
                </div>
              )}

            </div>

            {/* Quick-Test Sandbox */}
            <div className="bg-white dark:bg-[#1F2937] rounded-2xl shadow-md border border-gray-200 dark:border-slate-700/50 p-3 lg:p-4 transition-colors duration-300">
              <h4 className="text-[11px] font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-widest mb-2 flex items-center gap-2 transition-colors duration-300">
                <ImageIcon className="w-3.5 h-3.5" />
                No image? Try a sample:
              </h4>
              <div className="flex gap-2 lg:gap-3 overflow-x-auto pb-1 scrollbar-none snap-x">
                {SAMPLE_IMAGES.map((sample) => (
                  <button
                    key={sample.id}
                    onClick={() => selectSample(sample.url, sample.name)}
                    className="shrink-0 snap-start relative w-16 h-16 lg:w-20 lg:h-20 rounded-xl overflow-hidden border-2 border-gray-200 dark:border-slate-700 hover:border-[#10B981] dark:hover:border-[#10B981] transition-all group focus:outline-none focus:ring-2 focus:ring-[#10B981] focus:ring-offset-2 focus:ring-offset-white dark:focus:ring-offset-[#121824]"
                  >
                    <img src={sample.url} alt={sample.name} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent flex items-end p-1.5 opacity-0 group-hover:opacity-100 transition-opacity">
                      <span className="text-[9px] font-bold text-white uppercase tracking-wider leading-tight">{sample.name}</span>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Right Column: Real-Time Results & Actionable Intelligence */}
          <div id="results-panel" className={`bg-white dark:bg-[#1F2937] rounded-2xl shadow-xl dark:shadow-2xl border border-gray-200 dark:border-slate-700/50 flex flex-col overflow-hidden relative h-[360px] lg:h-[440px] w-full transition-colors duration-300 ${!imageSrc && status === 'idle' ? 'hidden lg:flex' : 'flex'}`}>
            {/* Minimalist Top Bar */}
            <div className="bg-gray-50 dark:bg-slate-800/50 border-b border-gray-200 dark:border-slate-700/50 px-4 lg:px-5 py-2.5 lg:py-3 flex items-center justify-between transition-colors duration-300">
              <span className="text-[10px] lg:text-xs font-mono text-slate-600 dark:text-slate-400 uppercase tracking-wider">Inference Results</span>
              {status === 'success' && <div className="w-2 h-2 rounded-full bg-[#10B981] shadow-[0_0_8px_rgba(16,185,129,0.6)] dark:shadow-[0_0_8px_rgba(16,185,129,1)]"></div>}
            </div>

            <div className="p-4 lg:p-6 flex flex-col flex-1 relative z-10 overflow-y-auto justify-center">

              {/* Empty State */}
              {status === 'idle' && !imageSrc && (
                <div className="flex-1 flex flex-col items-center justify-center text-center py-6">
                  <div className="w-14 h-14 rounded-full bg-gray-50 dark:bg-slate-800/50 flex items-center justify-center border border-gray-200 dark:border-slate-700/50 mb-4 relative transition-colors duration-300">
                    <Eye className="w-6 h-6 text-[#94A3B8] dark:text-slate-500" />
                    <div className="absolute inset-0 border border-gray-300 dark:border-slate-600 border-dashed rounded-full animate-[spin_10s_linear_infinite]"></div>
                  </div>
                  <p className="text-slate-600 dark:text-slate-300 font-light max-w-sm leading-relaxed transition-colors duration-300 text-xs lg:text-sm">
                    Awaiting image input... <br /><span className="text-slate-500 dark:text-slate-400 text-[11px]">Upload an image on the left to initialize AI analysis.</span>
                  </p>
                </div>
              )}

              {/* Loading Skeleton Panel */}
              {status === 'loading' && (
                <div className="flex-1 flex flex-col justify-center py-4 animate-pulse">
                  <div className="h-3 w-20 bg-gray-200 dark:bg-slate-800 rounded mb-2"></div>
                  <div className="h-6 w-1/2 bg-gray-200 dark:bg-slate-700 rounded mb-6"></div>

                  <div className="h-3 w-24 bg-gray-200 dark:bg-slate-800 rounded mb-2"></div>
                  <div className="w-full bg-gray-200 dark:bg-slate-800 rounded-full h-1.5 mb-6"></div>

                  <div className="bg-gray-100 dark:bg-slate-800/50 rounded-xl p-3 border border-gray-200 dark:border-slate-700/50 h-20"></div>
                </div>
              )}

              {/* Error State */}
              {status === 'error' && (
                <div className="bg-red-50 dark:bg-red-500/10 border border-red-200 dark:border-red-500/30 rounded-2xl p-4 lg:p-6 flex flex-col items-center justify-center text-center transition-colors duration-300">
                  <AlertCircle className="w-10 h-10 text-red-500 dark:text-red-400 mb-3" />
                  <p className="text-red-600 dark:text-red-300 font-medium text-xs lg:text-sm">{errorMessage}</p>
                  <button
                    onClick={clearWorkspace}
                    className="mt-4 px-4 py-1.5 bg-white hover:bg-gray-50 text-[#0F172A] border border-gray-200 dark:bg-slate-800 dark:hover:bg-slate-700 dark:border-slate-700 dark:text-white rounded-lg transition-colors text-xs font-semibold"
                  >
                    Try Again
                  </button>
                </div>
              )}

              {/* Success State */}
              {status === 'success' && result && (
                <div className="flex-1 flex flex-col animate-in fade-in slide-in-from-bottom-4 duration-500 justify-between">
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <p className="text-[#10B981] font-mono text-[10px] font-bold uppercase tracking-widest flex items-center gap-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        Prediction Complete
                      </p>
                    </div>
                    <h2 className="text-2xl lg:text-3xl xl:text-4xl font-black text-[#0F172A] dark:text-[#F9FAFB] uppercase tracking-tight break-words transition-colors duration-300">
                      {result.category}
                    </h2>
                  </div>

                  <div className="mt-4 mb-4">
                    <div className="flex justify-between items-end mb-1.5">
                      <span className="text-xs font-medium text-slate-600 dark:text-slate-400">Confidence Score</span>
                      <span className="text-sm font-bold font-mono text-[#0F172A] dark:text-[#F9FAFB]">{result.confidence}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-gray-100 dark:bg-slate-800 rounded-full overflow-hidden shadow-inner relative transition-colors duration-300">
                      <div
                        className={`absolute top-0 bottom-0 left-0 rounded-full transition-all duration-1000 ease-out flex items-center justify-end pr-1
                          ${result.confidence > 90 ? 'bg-[#10B981]' : result.confidence > 75 ? 'bg-[#A3E635]' : 'bg-[#F59E0B]'}`}
                        style={{ width: `${result.confidence}%` }}
                      >
                        <div className="w-1 h-1 bg-white/70 dark:bg-white/50 rounded-full shadow-[0_0_5px_white]"></div>
                      </div>
                    </div>
                  </div>

                  {/* Eco-Action Card */}
                  <div className="bg-gray-50 dark:bg-slate-800/60 backdrop-blur-md border border-gray-200 dark:border-slate-700/80 rounded-xl p-3.5 relative overflow-hidden group transition-colors duration-300">
                    <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-[#10B981] to-[#A3E635] dark:to-[#A3E635]"></div>

                    <div className="flex items-center justify-between gap-2 mb-3">
                      <h4 className="text-[11px] font-bold text-[#0F172A] dark:text-[#F9FAFB] uppercase tracking-wider flex items-center gap-1.5 transition-colors duration-300">
                        <Info className="w-3.5 h-3.5 text-[#10B981]" />
                        Sorting Directives
                      </h4>
                      <span className={`px-2 py-0.5 text-[9px] font-bold rounded-full uppercase tracking-wider border ${result.recyclable
                        ? 'bg-[#10B981]/10 text-[#10B981] border-[#10B981]/30 dark:bg-[#F59E0B]/10 dark:text-[#F59E0B] dark:border-[#F59E0B]/30'
                        : 'bg-red-100 text-red-600 border-red-200 dark:bg-red-500/10 dark:text-red-500 dark:border-red-500/30'
                        }`}>
                        {result.recyclable ? 'Recyclable' : 'Non-Recyclable'}
                      </span>
                    </div>

                    <div className="space-y-2.5">
                      <div className="flex gap-2.5 items-start">
                        <div className="w-7 h-7 rounded-lg bg-white dark:bg-slate-900 border border-gray-200 dark:border-slate-700 flex flex-col justify-center items-center text-slate-600 dark:text-slate-400 shrink-0 transition-colors duration-300">
                          <Trash className="w-3.5 h-3.5" />
                        </div>
                        <div>
                          <p className="text-[9px] text-[#94A3B8] dark:text-slate-500 uppercase tracking-widest font-semibold">Target Bin</p>
                          <p className="text-xs text-[#0F172A] dark:text-[#F9FAFB] font-semibold transition-colors duration-300">{result.binType}</p>
                        </div>
                      </div>

                      <div className="bg-white dark:bg-slate-900/50 rounded-lg p-2.5 border border-gray-200 dark:border-slate-700/50 transition-colors duration-300">
                        <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed transition-colors duration-300">
                          <span className="text-[#0F172A] dark:text-[#F9FAFB] font-semibold">Instruction:</span> {result.tips}
                        </p>
                      </div>
                    </div>
                  </div>

                </div>
              )}
            </div>

            {/* Background Decoration */}
            {status === 'success' && (
              <div className="absolute -bottom-32 -right-32 w-96 h-96 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-[#10B981]/10 to-transparent blur-3xl pointer-events-none"></div>
            )}

          </div>
        </div>

      </div>

      {/* Global CSS for Animations */}
      <style>{`
        @keyframes scan {
          0%, 100% { top: 0%; opacity: 0; }
          10% { opacity: 1; }
          90% { opacity: 1; }
          50% { top: 100%; }
        }
        @keyframes progress {
          0% { width: 0%; transform: translateX(-100%); }
          50% { width: 100%; transform: translateX(0); }
          100% { width: 0%; transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
}

// Minimal missing icon wrapper
function CpuIcon(props: any) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect width="16" height="16" x="4" y="4" rx="2" />
      <rect width="6" height="6" x="9" y="9" rx="1" />
      <path d="M15 2v2" />
      <path d="M15 20v2" />
      <path d="M2 15h2" />
      <path d="M2 9h2" />
      <path d="M20 15h2" />
      <path d="M20 9h2" />
      <path d="M9 2v2" />
      <path d="M9 20v2" />
    </svg>
  )
}

