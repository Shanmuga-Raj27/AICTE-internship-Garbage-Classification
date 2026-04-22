import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Classifier from '../views/Classifier.vue'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: Home
  },
  {
    path: '/classify',
    name: 'Classifier',
    component: Classifier
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
