// Main JavaScript for CS:GO ML Predictor

// Global configuration
const CONFIG = {
    API_BASE_URL: '',
    ANIMATION_DURATION: 300,
    DEBOUNCE_DELAY: 500
};

// Utility functions
const Utils = {
    // Debounce function for input validation
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Format numbers for display
    formatNumber: function(num, decimals = 2) {
        return Number(num).toFixed(decimals);
    },

    // Format percentage
    formatPercentage: function(num, decimals = 1) {
        return (Number(num) * 100).toFixed(decimals) + '%';
    },

    // Show notification
    showNotification: function(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} notification fade-in`;
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            min-width: 300px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        `;
        notification.innerHTML = `
            <button type="button" class="btn-close float-end" aria-label="Close"></button>
            <div>${message}</div>
        `;

        document.body.appendChild(notification);

        // Close button functionality
        notification.querySelector('.btn-close').addEventListener('click', () => {
            notification.remove();
        });

        // Auto remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, duration);
        }
    },

    // Validate form inputs
    validateInput: function(input, min, max, required = true) {
        const value = parseFloat(input.value);
        const errors = [];

        if (required && (input.value === '' || isNaN(value))) {
            errors.push('Este campo es requerido');
        }

        if (!isNaN(value)) {
            if (value < min) {
                errors.push(`El valor debe ser mayor o igual a ${min}`);
            }
            if (value > max) {
                errors.push(`El valor debe ser menor o igual a ${max}`);
            }
        }

        return errors;
    },

    // Show input validation errors
    showInputErrors: function(input, errors) {
        // Remove existing error messages
        const existingError = input.parentNode.querySelector('.invalid-feedback');
        if (existingError) {
            existingError.remove();
        }

        if (errors.length > 0) {
            input.classList.add('is-invalid');
            const errorDiv = document.createElement('div');
            errorDiv.className = 'invalid-feedback';
            errorDiv.textContent = errors[0];
            input.parentNode.appendChild(errorDiv);
        } else {
            input.classList.remove('is-invalid');
            input.classList.add('is-valid');
        }
    },

    // Clear all validation states
    clearValidation: function(form) {
        const inputs = form.querySelectorAll('.form-control');
        inputs.forEach(input => {
            input.classList.remove('is-invalid', 'is-valid');
            const errorDiv = input.parentNode.querySelector('.invalid-feedback');
            if (errorDiv) {
                errorDiv.remove();
            }
        });
    },

    // Animate number counting
    animateNumber: function(element, start, end, duration = 1000) {
        const startTime = performance.now();
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const easeOutCubic = 1 - Math.pow(1 - progress, 3);
            const current = start + (end - start) * easeOutCubic;
            
            element.textContent = Utils.formatNumber(current, 1);
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        requestAnimationFrame(animate);
    },

    // Copy text to clipboard
    copyToClipboard: function(text) {
        if (navigator.clipboard && window.isSecureContext) {
            return navigator.clipboard.writeText(text);
        } else {
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = text;
            textArea.style.position = 'fixed';
            textArea.style.left = '-999999px';
            textArea.style.top = '-999999px';
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            return new Promise((resolve, reject) => {
                if (document.execCommand('copy')) {
                    textArea.remove();
                    resolve();
                } else {
                    textArea.remove();
                    reject();
                }
            });
        }
    }
};

// API functions
const API = {
    // Make API request
    request: async function(endpoint, options = {}) {
        const defaultOptions = {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        };

        const config = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(CONFIG.API_BASE_URL + endpoint, config);
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('API request failed:', error);
            throw error;
        }
    },

    // Get model information
    getModelInfo: function() {
        return this.request('/api/model_info');
    },

    // Predict effectiveness
    predictEffectiveness: function(data) {
        return this.request('/api/predict/effectiveness', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    },

    // Predict match kills
    predictMatchKills: function(data) {
        return this.request('/api/predict/match_kills', {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }
};

// Form validation setup
const FormValidation = {
    setupValidation: function() {
        // Classification form validation
        const classificationForm = document.getElementById('classificationForm');
        if (classificationForm) {
            this.setupClassificationValidation(classificationForm);
        }

        // Regression form validation
        const regressionForm = document.getElementById('regressionForm');
        if (regressionForm) {
            this.setupRegressionValidation(regressionForm);
        }
    },

    setupClassificationValidation: function(form) {
        const roundHeadshots = form.querySelector('#roundHeadshots');
        const grenadeEffectiveness = form.querySelector('#grenadeEffectiveness');

        if (roundHeadshots) {
            const validateHeadshots = Utils.debounce(() => {
                const errors = Utils.validateInput(roundHeadshots, 0, 10);
                Utils.showInputErrors(roundHeadshots, errors);
            }, CONFIG.DEBOUNCE_DELAY);

            roundHeadshots.addEventListener('input', validateHeadshots);
            roundHeadshots.addEventListener('blur', validateHeadshots);
        }

        if (grenadeEffectiveness) {
            const validateGrenades = Utils.debounce(() => {
                const errors = Utils.validateInput(grenadeEffectiveness, 0, 20);
                Utils.showInputErrors(grenadeEffectiveness, errors);
            }, CONFIG.DEBOUNCE_DELAY);

            grenadeEffectiveness.addEventListener('input', validateGrenades);
            grenadeEffectiveness.addEventListener('blur', validateGrenades);
        }
    },

    setupRegressionValidation: function(form) {
        const inputs = {
            roundKills: { min: 0, max: 5 },
            matchHeadshots: { min: 0, max: 50 },
            teamEquipmentValue: { min: 1000, max: 30000 },
            matchAssists: { min: 0, max: 30 }
        };

        Object.entries(inputs).forEach(([id, constraints]) => {
            const input = form.querySelector(`#${id}`);
            if (input) {
                const validate = Utils.debounce(() => {
                    const errors = Utils.validateInput(input, constraints.min, constraints.max);
                    Utils.showInputErrors(input, errors);
                }, CONFIG.DEBOUNCE_DELAY);

                input.addEventListener('input', validate);
                input.addEventListener('blur', validate);
            }
        });
    }
};

// Theme management
const ThemeManager = {
    init: function() {
        // Check for saved theme preference or default to 'light'
        const currentTheme = localStorage.getItem('theme') || 'light';
        this.setTheme(currentTheme);
        
        // Create theme toggle button if it doesn't exist
        this.createThemeToggle();
    },

    setTheme: function(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('theme', theme);
        
        // Update theme toggle button
        const themeToggle = document.getElementById('theme-toggle');
        if (themeToggle) {
            const icon = themeToggle.querySelector('i');
            if (theme === 'dark') {
                icon.className = 'fas fa-sun';
            } else {
                icon.className = 'fas fa-moon';
            }
        }
    },

    toggleTheme: function() {
        const currentTheme = localStorage.getItem('theme') || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        this.setTheme(newTheme);
    },

    createThemeToggle: function() {
        // Only create if it doesn't exist
        if (document.getElementById('theme-toggle')) return;

        const navbar = document.querySelector('.navbar-nav');
        if (navbar) {
            const themeToggleHTML = `
                <li class="nav-item">
                    <button class="nav-link btn btn-link" id="theme-toggle" title="Toggle theme">
                        <i class="fas fa-moon"></i>
                    </button>
                </li>
            `;
            navbar.insertAdjacentHTML('beforeend', themeToggleHTML);
            
            document.getElementById('theme-toggle').addEventListener('click', () => {
                this.toggleTheme();
            });
        }
    }
};

// Performance monitoring
const Performance = {
    startTime: null,
    
    start: function(label) {
        this.startTime = performance.now();
        console.log(`Starting ${label}...`);
    },
    
    end: function(label) {
        if (this.startTime) {
            const endTime = performance.now();
            const duration = endTime - this.startTime;
            console.log(`${label} completed in ${duration.toFixed(2)}ms`);
            this.startTime = null;
            return duration;
        }
    }
};

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('CS:GO ML Predictor - Initializing...');
    
    // Initialize components
    FormValidation.setupValidation();
    ThemeManager.init();
    
    // Add smooth scrolling to all anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add loading states to buttons
    document.querySelectorAll('button[type="submit"]').forEach(button => {
        button.addEventListener('click', function() {
            const originalText = this.innerHTML;
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Procesando...';
            this.disabled = true;
            
            // Re-enable after 10 seconds as fallback
            setTimeout(() => {
                this.innerHTML = originalText;
                this.disabled = false;
            }, 10000);
        });
    });
    
    // Add tooltips to elements with title attribute
    const tooltipElements = document.querySelectorAll('[title]');
    tooltipElements.forEach(element => {
        // Basic tooltip functionality
        element.addEventListener('mouseenter', function() {
            const tooltip = document.createElement('div');
            tooltip.className = 'custom-tooltip';
            tooltip.textContent = this.title;
            tooltip.style.cssText = `
                position: absolute;
                background: #333;
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
                font-size: 12px;
                z-index: 10000;
                pointer-events: none;
                opacity: 0.9;
            `;
            document.body.appendChild(tooltip);
            
            // Position tooltip
            const rect = this.getBoundingClientRect();
            tooltip.style.left = rect.left + (rect.width / 2) - (tooltip.offsetWidth / 2) + 'px';
            tooltip.style.top = rect.top - tooltip.offsetHeight - 5 + 'px';
            
            this._tooltip = tooltip;
        });
        
        element.addEventListener('mouseleave', function() {
            if (this._tooltip) {
                this._tooltip.remove();
                this._tooltip = null;
            }
        });
    });
    
    console.log('CS:GO ML Predictor - Initialization complete');
});

// Export utilities for use in other scripts
window.CSGOMLPredictor = {
    Utils,
    API,
    ThemeManager,
    Performance
};