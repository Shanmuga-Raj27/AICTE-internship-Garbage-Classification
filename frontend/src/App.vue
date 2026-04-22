<script setup>
import { RouterLink, RouterView } from 'vue-router'
import { Recycle, Scan } from 'lucide-vue-next'
</script>

<template>
  <div class="min-h-screen bg-gray-50 text-gray-900 font-sans flex flex-col">
    <!-- Navigation Bar -->
    <nav class="bg-white/80 backdrop-blur-md shadow-sm sticky top-0 z-50 border-b border-gray-200 text-gray-800">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16 items-center">
          <!-- Logo -->
          <div class="flex items-center space-x-2">
            <Recycle class="h-8 w-8 text-emerald-500" />
            <span class="font-bold text-xl tracking-tight text-gray-900">EcoClassify API</span>
          </div>

          <!-- Links -->
          <div class="flex space-x-6 items-center">
            <RouterLink 
              to="/" 
              class="font-medium text-gray-600 hover:text-emerald-600 transition-colors duration-200"
              active-class="text-emerald-600 font-semibold"
            >
              Info
            </RouterLink>
            <RouterLink 
              to="/classify"
              class="flex items-center space-x-1.5 font-medium bg-emerald-500 hover:bg-emerald-600 text-white px-4 py-2 rounded-full shadow-md shadow-emerald-500/20 transition-all duration-200 transform hover:-translate-y-0.5"
              active-class="ring-2 ring-emerald-500 ring-offset-2"
            >
              <Scan class="w-4 h-4" />
              <span>Classify</span>
            </RouterLink>
          </div>
        </div>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="flex-grow w-full">
      <RouterView v-slot="{ Component }">
        <transition name="fade" mode="out-in">
          <component :is="Component" />
        </transition>
      </RouterView>
    </main>
    
    <!-- Footer -->
    <footer class="bg-white border-t border-gray-200 py-6 mt-12 text-center text-sm text-gray-500">
      <p>Educational Platform for Waste Management & AI Classification</p>
    </footer>
  </div>
</template>

<style>
.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.3s ease, transform 0.3s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
  transform: translateY(10px);
}
</style>
