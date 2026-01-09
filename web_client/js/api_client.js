/**
 * API Client for ShioRIS3 Web Server
 * Handles all communication with the backend
 */

export class ApiClient {
    constructor(baseUrl = '') {
        // If baseUrl is empty, use current origin
        this.baseUrl = baseUrl || window.location.origin;
        console.log('API Client initialized with base URL:', this.baseUrl);
    }

    /**
     * Make a GET request to the API
     */
    async get(endpoint, timeout = 30000) {
        const url = `${this.baseUrl}${endpoint}`;
        console.log('GET request:', url);

        try {
            // Create abort controller for timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), timeout);

            const response = await fetch(url, {
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Log response size
            const contentLength = response.headers.get('content-length');
            if (contentLength) {
                console.log(`Response size: ${(parseInt(contentLength) / 1024).toFixed(2)} KB`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            if (error.name === 'AbortError') {
                console.error('Request timeout after', timeout, 'ms');
                throw new Error(`Request timeout after ${timeout}ms`);
            }
            console.error('API request failed:', error);
            throw error;
        }
    }

    /**
     * Make a POST request to the API
     */
    async post(endpoint, data) {
        const url = `${this.baseUrl}${endpoint}`;
        console.log('POST request:', url, data);

        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const responseData = await response.json();
            return responseData;
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    }

    /**
     * Get list of patients
     */
    async getPatients() {
        return await this.get('/api/patients');
    }

    /**
     * Get volume data for a patient
     */
    async getVolume(patientId) {
        return await this.get(`/api/volume/${patientId}`);
    }

    /**
     * Get structure data for a patient
     * @param {string} patientId - Patient ID
     * @param {number} simplify - Simplification level (1=none, 2=half points, 3=third, etc.)
     */
    async getStructures(patientId, simplify = 1) {
        const url = simplify > 1
            ? `/api/structures/${patientId}?simplify=${simplify}`
            : `/api/structures/${patientId}`;
        return await this.get(url, 60000);  // Use longer timeout
    }

    /**
     * Get dose data for a patient
     */
    async getDose(patientId) {
        // Use longer timeout for dose data (can be large)
        return await this.get(`/api/dose/${patientId}`, 60000);
    }

    /**
     * Get a specific slice image as a blob with optional window/level
     */
    async getSliceImage(patientId, orientation, sliceIndex, window = null, level = null) {
        // Add timestamp to prevent caching
        const timestamp = Date.now();
        let url = `${this.baseUrl}/api/slice/${patientId}/${orientation}/${sliceIndex}?t=${timestamp}`;

        // Add window/level query parameters if provided
        if (window !== null && window !== undefined) {
            url += `&window=${encodeURIComponent(window)}`;
        }
        if (level !== null && level !== undefined) {
            url += `&level=${encodeURIComponent(level)}`;
        }

        console.log('GET slice image:', url);

        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const blob = await response.blob();
            return blob;
        } catch (error) {
            console.error('Slice image request failed:', error);
            throw error;
        }
    }

    /**
     * Get a specific slice image as a data URL with optional window/level
     */
    async getSliceImageURL(patientId, orientation, sliceIndex, window = null, level = null) {
        const blob = await this.getSliceImage(patientId, orientation, sliceIndex, window, level);
        return URL.createObjectURL(blob);
    }

    /**
     * Test connection to server
     */
    async testConnection() {
        try {
            const data = await this.getPatients();
            console.log('Connection test successful:', data);
            return true;
        } catch (error) {
            console.error('Connection test failed:', error);
            return false;
        }
    }
}

// Create a singleton instance
export const apiClient = new ApiClient();
