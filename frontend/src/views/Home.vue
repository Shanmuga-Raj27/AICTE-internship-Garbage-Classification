<script setup>
import { ref, onMounted } from 'vue'
import { Info, CheckCircle2, XCircle } from 'lucide-vue-next'

const categories = ref([])
const loading = ref(true)
const error = ref(null)

onMounted(async () => {
  try {
    const response = await fetch('http://localhost:8000/categories')
    if (!response.ok) throw new Error('Failed to fetch categories')
    const data = await response.json()
    categories.value = data.categories
  } catch (err) {
    error.value = "Unable to connect to the backend API. Showing offline data."
    categories.value = [
      { name: "Cardboard", description: "Boxes, packaging", icon: "📦", recyclable: true, tips: "Flatten before recycling." },
      { name: "Glass", description: "Bottles, jars", icon: "🍶", recyclable: true, tips: "Rinse clean." },
      { name: "Metal", description: "Cans, foil", icon: "🥫", recyclable: true, tips: "Rinse food out." },
      { name: "Paper", description: "Newspapers", icon: "📄", recyclable: true, tips: "Keep dry." },
      { name: "Plastic", description: "Bottles, bags", icon: "🧴", recyclable: true, tips: "Check recycle number." },
      { name: "Trash", description: "Non-recyclable", icon: "🗑️", recyclable: false, tips: "Goes to landfill." }
    ]
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
    <div class="text-center max-w-3xl mx-auto mb-16">
      <h1 class="text-4xl md:text-5xl font-extrabold text-gray-900 tracking-tight mb-4">
        Know Your <span class="bg-clip-text text-transparent bg-gradient-to-r from-emerald-500 to-teal-400">Waste</span>
      </h1>
      <p class="text-lg text-gray-600">
        Our AI can classify waste into 6 categories. Learn how to sort them properly and contribute to a greener planet.
      </p>
    </div>

    <!-- Error state -->
    <div v-if="error" class="mb-8 p-4 bg-amber-50 border-l-4 border-amber-400 rounded-r-md flex items-start space-x-3 text-amber-800">
      <Info class="w-5 h-5 flex-shrink-0 mt-0.5" />
      <p>{{ error }}</p>
    </div>

    <!-- Loading state -->
    <div v-if="loading" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
      <div v-for="i in 6" :key="i" class="bg-white rounded-2xl shadow-sm border border-gray-100 p-6 animate-pulse">
        <div class="w-16 h-16 bg-gray-200 rounded-2xl mb-4"></div>
        <div class="h-6 bg-gray-200 rounded w-1/2 mb-3"></div>
        <div class="h-4 bg-gray-200 rounded w-full mb-2"></div>
        <div class="h-4 bg-gray-200 rounded w-2/3"></div>
      </div>
    </div>

    <!-- Categories Grid -->
    <div v-else class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
      <div 
        v-for="cat in categories" 
        :key="cat.name"
        class="group bg-white rounded-2xl shadow-sm hover:shadow-xl border border-gray-100 hover:border-emerald-100 p-8 transition-all duration-300 transform hover:-translate-y-1 relative overflow-hidden"
      >
        <!-- Background blob decoration -->
        <div class="absolute -right-8 -top-8 w-32 h-32 bg-emerald-50 rounded-full blur-3xl group-hover:bg-emerald-100 transition-colors duration-500 z-0"></div>

        <div class="relative z-10 flex flex-col h-full">
          <div class="flex items-center justify-between mb-4">
            <div class="text-5xl drop-shadow-sm group-hover:scale-110 transition-transform duration-300 transform-origin-center">
              {{ cat.icon }}
            </div>
            <div 
              class="px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider flex items-center space-x-1"
              :class="cat.recyclable ? 'bg-emerald-100 text-emerald-700' : 'bg-gray-100 text-gray-700'"
            >
              <CheckCircle2 v-if="cat.recyclable" class="w-4 h-4" />
              <XCircle v-else class="w-4 h-4" />
              <span>{{ cat.recyclable ? 'Recyclable' : 'Trash' }}</span>
            </div>
          </div>
          
          <h3 class="text-2xl font-bold text-gray-900 mb-2">{{ cat.name }}</h3>
          <p class="text-gray-500 mb-4 font-medium">{{ cat.description }}</p>
          
          <div class="mt-auto pt-4 border-t border-gray-100">
            <p class="text-sm text-gray-600">
              <span class="font-semibold text-gray-900 block mb-1">Eco Tip:</span>
              {{ cat.tips }}
            </p>
          </div>
        </div>
      </div>
    </div>

  </div>
</template>
