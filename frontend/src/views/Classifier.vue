<script setup>
import { ref, watch } from 'vue'
import { UploadCloud, Image as ImageIcon, Loader2, ArrowRight, RefreshCw, XCircle, Scan } from 'lucide-vue-next'

const fileInput = ref(null)
const previewUrl = ref(null)
const selectedFile = ref(null)
const isDragging = ref(false)
const isPredicting = ref(false)
const result = ref(null)
const error = ref(null)

const handleDragOver = (e) => {
  e.preventDefault()
  isDragging.value = true
}

const handleDragLeave = (e) => {
  e.preventDefault()
  isDragging.value = false
}

const handleDrop = (e) => {
  e.preventDefault()
  isDragging.value = false
  if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    processFile(e.dataTransfer.files[0])
  }
}

const triggerFileInput = () => {
  fileInput.value.click()
}

const handleFileSelect = (e) => {
  if (e.target.files && e.target.files.length > 0) {
    processFile(e.target.files[0])
  }
}

const processFile = (file) => {
  error.value = null
  result.value = null

  // Validate type
  const allowed = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp']
  if (!allowed.includes(file.type)) {
    error.value = "Please upload a valid image file (JPEG, PNG, GIF, BMP)."
    return
  }

  // Preview
  selectedFile.value = file
  previewUrl.value = URL.createObjectURL(file)
}

const runPrediction = async () => {
  if (!selectedFile.value) return

  isPredicting.value = true
  error.value = null

  const formData = new FormData()
  formData.append('file', selectedFile.value)

  try {
    const response = await fetch(`${import.meta.env.VITE_API_URL}/predict`, {
      method: 'POST',
      body: formData
    })
    
    const data = await response.json()
    if (!response.ok) {
      throw new Error(data.detail || data.error || 'Prediction failed')
    }

    result.value = data
  } catch (err) {
    error.value = err.message
  } finally {
    isPredicting.value = false
  }
}

const reset = () => {
  selectedFile.value = null
  previewUrl.value = null
  result.value = null
  error.value = null
  if (fileInput.value) fileInput.value.value = ''
}

// Ensure URL is revoked to prevent memory leaks
watch(previewUrl, (newUrl, oldUrl) => {
  if (oldUrl) URL.revokeObjectURL(oldUrl)
})
</script>

<template>
  <div class="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12 flex flex-col items-center">
    
    <div class="text-center mb-10 w-full max-w-2xl">
      <h2 class="text-3xl font-bold text-gray-900 mb-3">AI Waste Classification</h2>
      <p class="text-gray-500">Upload an image of a waste item, and our AI will identify it and tell you how to dispose of it properly.</p>
    </div>

    <!-- Main Card -->
    <div class="w-full bg-white rounded-3xl shadow-xl shadow-gray-200/50 border border-gray-100 overflow-hidden relative">
      
      <div class="grid grid-cols-1 md:grid-cols-2">
        
        <!-- Left half: Upload section -->
        <div class="p-8 md:p-10 md:border-r border-gray-100 flex flex-col h-full bg-gray-50/50">
          <h3 class="text-lg font-semibold text-gray-700 mb-6 flex items-center">
            📝 Image Source
          </h3>
          
          <div 
            class="flex-grow border-2 border-dashed rounded-2xl flex flex-col items-center justify-center p-8 transition-colors duration-200 text-center cursor-pointer min-h-[300px]"
            :class="[
              isDragging ? 'border-emerald-500 bg-emerald-50' : 'border-gray-300 bg-white hover:bg-gray-50',
              previewUrl ? 'border-none p-2 bg-black/5' : ''
            ]"
            @dragover="handleDragOver"
            @dragleave="handleDragLeave"
            @drop="handleDrop"
            @click="!previewUrl && triggerFileInput()"
          >
            <!-- No file selected -->
            <template v-if="!previewUrl">
              <div class="w-20 h-20 bg-emerald-100 text-emerald-600 rounded-full flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <UploadCloud class="w-10 h-10" />
              </div>
              <p class="text-gray-700 font-medium text-lg mb-1">Drag & drop image here</p>
              <p class="text-sm text-gray-400">or click to browse from device</p>
            </template>
            
            <!-- File selected -->
            <template v-else>
              <div class="relative w-full h-full rounded-xl overflow-hidden group">
                <img :src="previewUrl" class="w-full h-full object-cover rounded-xl" alt="Preview"/>
                
                <!-- Overlay actions -->
                <div class="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center space-x-4 backdrop-blur-sm">
                  <button @click.stop="triggerFileInput" class="bg-white/90 text-gray-900 p-2 rounded-full hover:bg-white tooltip" title="Change Image">
                    <ImageIcon class="w-5 h-5"/>
                  </button>
                  <button @click.stop="reset" class="bg-red-500/90 text-white p-2 rounded-full hover:bg-red-500 tooltip" title="Remove">
                    <XCircle class="w-5 h-5"/>
                  </button>
                </div>
              </div>
            </template>
          </div>

          <input 
            type="file" 
            ref="fileInput" 
            class="hidden" 
            accept="image/jpeg,image/png,image/gif,image/bmp" 
            @change="handleFileSelect"
          >

          <button 
            v-if="selectedFile && !result"
            @click="runPrediction"
            :disabled="isPredicting"
            class="mt-6 w-full py-4 rounded-xl font-bold flex items-center justify-center space-x-2 transition-all transform hover:-translate-y-0.5 shadow-lg"
            :class="isPredicting ? 'bg-emerald-400 text-white cursor-not-allowed shadow-emerald-400/20' : 'bg-emerald-500 hover:bg-emerald-600 text-white shadow-emerald-500/30'"
          >
            <Loader2 v-if="isPredicting" class="w-5 h-5 animate-spin" />
            <Scan v-else class="w-5 h-5" />
            <span>{{ isPredicting ? 'Analyzing Image...' : 'Classify Waste' }}</span>
          </button>
        </div>

        <!-- Right half: Result section -->
        <div class="p-8 md:p-10 bg-white flex flex-col h-full relative">
          <h3 class="text-lg font-semibold text-gray-700 mb-6 border-b pb-4">
            Analysis Results
          </h3>

          <!-- Error Message -->
          <div v-if="error" class="bg-red-50 text-red-700 p-4 rounded-xl flex items-start space-x-3 text-sm">
            <XCircle class="w-5 h-5 flex-shrink-0 mt-0.5" />
            <span>{{ error }}</span>
          </div>

          <!-- Waiting State -->
          <div v-else-if="!result" class="flex-grow flex flex-col items-center justify-center text-center opacity-50">
             <div class="w-24 h-24 mb-4 opacity-20 filter grayscale rounded-full border-4 border-dashed border-gray-400 flex items-center justify-center">
                 <Loader2 v-if="isPredicting" class="w-10 h-10 animate-spin text-emerald-500"/>
                 <span v-else class="text-4xl">🤖</span>
             </div>
             <p class="font-medium text-gray-500">
               {{ isPredicting ? 'Processing AI model paths...' : 'Upload an image and hit Classify to see results here.' }}
             </p>
          </div>

          <!-- Result Data -->
          <div v-else class="flex-grow flex flex-col items-center animate-fade-in-up">
            
            <div class="text-8xl mb-4 drop-shadow-md animate-bounce">
              {{ result.category_info?.icon || '🔍' }}
            </div>

            <div class="text-sm font-bold tracking-widest text-emerald-500 uppercase mb-1">Detected Object</div>
            <h4 class="text-4xl font-extrabold text-gray-900 mb-2">{{ result.prediction }}</h4>
            
            <div class="bg-gray-100 rounded-full px-4 py-1.5 mb-8 flex items-center space-x-2 text-sm font-semibold">
              <span class="text-gray-500">Confidence:</span>
              <span class="text-emerald-600">{{ result.confidence_percentage }}</span>
            </div>

            <!-- Context Card -->
            <div class="w-full bg-emerald-50 rounded-2xl p-6 border border-emerald-100 relative overflow-hidden">
               <!-- decorative shape -->
               <div class="absolute -right-6 -top-6 w-24 h-24 bg-emerald-200/50 rounded-full blur-xl"></div>
               
               <div class="relative z-10">
                 <div class="font-semibold text-emerald-800 flex items-center mb-2">
                    <span class="mr-2">💡 Tips</span>
                 </div>
                 <p class="text-emerald-700/80 text-sm leading-relaxed mb-4">
                   {{ result.category_info?.tips || 'N/A' }}
                 </p>

                 <div v-if="result.category_info" class="pt-4 border-t border-emerald-200 flex justify-between items-center text-sm">
                    <span class="text-emerald-800 font-medium">Recyclable?</span>
                    <span :class="result.category_info.recyclable ? 'text-green-600' : 'text-red-600'" class="font-bold">
                       {{ result.category_info.recyclable ? '✅ Yes' : '❌ No' }}
                    </span>
                 </div>
               </div>
            </div>

            <button @click="reset" class="mt-8 text-sm font-semibold text-gray-500 hover:text-gray-800 flex items-center space-x-1.5 transition-colors">
              <RefreshCw class="w-4 h-4"/>
              <span>Scan another item</span>
            </button>
          </div>

        </div>

      </div>
    </div>
  </div>
</template>

<style scoped>
.animate-fade-in-up {
  animation: fadeInUp 0.5s ease-out forwards;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>
