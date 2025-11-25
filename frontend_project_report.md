# Dog Breed Identifier - Frontend Project Report

## 1. Introduction
The **Dog Breed Identifier** is a modern web application designed to help users identify dog breeds from images. Leveraging the power of machine learning (via a backend API) and a responsive frontend interface, users can upload photos or use their device's camera to get instant breed predictions. The application also provides detailed information about each breed, including their country of origin displayed on an interactive map.

## 2. Technology Stack

The frontend is built using a modern, type-safe stack designed for performance, scalability, and a premium user experience.

### Core Framework & Language
*   **Framework**: **[Next.js 15](https://nextjs.org/)** (App Router)
    *   *Role*: The React framework for production.
    *   *Why*: Provides server-side rendering (SSR) for SEO, efficient routing, and optimized build performance.
*   **Language**: **[TypeScript](https://www.typescriptlang.org/)**
    *   *Role*: Static typing for JavaScript.
    *   *Why*: Ensures code quality, reduces runtime errors, and improves developer productivity with better tooling support.

### UI & Styling
*   **Styling Engine**: **[Tailwind CSS 4](https://tailwindcss.com/)**
    *   *Role*: Utility-first CSS framework.
    *   *Why*: Enables rapid UI development with a consistent design system and built-in responsiveness.
*   **Icons**: **[React Icons](https://react-icons.github.io/react-icons/)**
    *   *Role*: Icon library.
    *   *Why*: Provides a comprehensive collection of icons from various libraries (Fa, Md, Io) with tree-shaking support.

### State Management & Data Fetching
*   **State Management**: **[Zustand](https://github.com/pmndrs/zustand)**
    *   *Role*: Global state management store.
    *   *Why*: Lightweight, unopinionated, and hook-based, making it perfect for managing the prediction cache and UI state without the boilerplate of Redux.
*   **HTTP Client**: **[Axios](https://axios-http.com/)**
    *   *Role*: Promise-based HTTP client.
    *   *Why*: Simplifies making asynchronous requests to the backend API with built-in interceptors and error handling.

### Maps & Visualization
*   **Mapping Library**: **[Leaflet](https://leafletjs.com/)** & **[React-Leaflet](https://react-leaflet.js.org/)**
    *   *Role*: Interactive maps.
    *   *Why*: Lightweight open-source library for rendering the breed origin map with custom markers and dynamic zooming.

### Cloud & Storage
*   **Object Storage**: **AWS SDK** (configured for **Tebi**)
    *   *Role*: Cloud storage interface.
    *   *Why*: Used for securely uploading user images via S3-compatible APIs, offloading storage requirements from the application server.

## 3. Project Structure
The project follows the standard Next.js App Router structure:

*   **`src/app`**: Contains the application routes and pages.
    *   `page.tsx`: The main entry point (Home page).
    *   `layout.tsx`: Defines the global layout (fonts, metadata).
    *   `api/`: Backend proxy routes for prediction and upload handling.
*   **`src/components`**: Reusable UI components.
    *   `BreedSearchBar.tsx`: Search input with autocomplete and voice support.
    *   `BreedMap.tsx`: Interactive map component.
    *   `CameraModal.tsx`: Modal for camera capture and file upload.
    *   `MainContent.tsx`: Orchestrates the main view logic.
*   **`src/lib`**: Core logic and helper functions.
    *   `upload.ts`: Handles file hashing and S3 uploads.
    *   `prediction-store.ts`: Zustand store for caching predictions.
    *   `breed-utils.ts`: Helpers for fetching breed information.

## 4. Key Features

### 🔍 AI-Powered Breed Identification
Users can upload an image or take a photo to identify the dog breed. The system calculates a unique file hash to check for cached results before sending the image to the backend for analysis.

### 🎙️ Voice & Text Search
A powerful search bar allows users to find breeds by name. It features:
*   **Autocomplete**: Real-time suggestions based on available breeds.
*   **Voice Search**: Integrated microphone support for hands-free searching (Chrome-supported).

### 🗺️ Interactive Origin Map
When a breed is identified or selected, an interactive map highlights its country of origin. This feature uses GeoJSON data to render countries and dynamically zooms to the relevant location.

### 📱 Responsive & Premium Design
The UI is designed with a "mobile-first" approach, ensuring it looks great on all devices. It features glassmorphism effects, smooth transitions, and a clean, modern aesthetic.

## 5. Component Overview

### `BreedSearchBar`
This component is the central hub for user interaction. It manages the search state, handles voice input using the Web Speech API, and integrates the file drop zone for drag-and-drop uploads.

### `BreedMap`
Built with `react-leaflet`, this component visualizes the geographical origin of the dog breeds. It dynamically updates the map view to center on the origin country and applies custom styling to highlight it.

### `MainContent`
This parent component manages the flow of data between the search bar, the prediction results, and the map. It handles the loading states and error management for a seamless user experience.

## 6. Technical Implementation Details

### Efficient File Uploads
To optimize performance and security, the application uses a **Presigned URL** pattern.
1.  The client calculates a SHA-256 hash of the file.
2.  It requests a secure upload URL from the backend.
3.  The file is uploaded directly to the object storage (Tebi/S3), bypassing the application server to reduce load.

### State Management & Caching
**Zustand** is used to manage global state and cache prediction results. This means if a user uploads the same image twice, the result is retrieved instantly from the client-side cache without a network request.

## 7. Future Enhancements
*   **PWA Support**: Turn the website into a Progressive Web App for offline capabilities.
*   **Community Features**: Allow users to save their favorite breeds or share results on social media.
*   **Multi-Language Support**: Localize the interface for a global audience.
*   **Compare Breeds**: A feature to compare two dog breeds side-by-side.
