// Auth functions
function checkAuth() {
    const user = JSON.parse(localStorage.getItem('currentUser'));
    const currentPage = window.location.pathname.split('/').pop();

    // If user is not logged in
    if (!user) {
        // Allow access only to register.html and login.html
        if (currentPage !== 'register.html' && currentPage !== 'login.html') {
            window.location.href = 'login.html';
            return null;
        }
    } else {
        // User is logged in
        if (currentPage === 'register.html' || currentPage === 'login.html') {
            // Redirect to index.html if trying to access login or register pages while logged in
            window.location.href = 'index.html';
            return user;
        }
    }

    return user;
}

// Call checkAuth on every page load
document.addEventListener('DOMContentLoaded', checkAuth);

// Handle Register Form
const registerForm = document.querySelector('.register-form');
if (registerForm) {
    registerForm.addEventListener('submit', (e) => {
        e.preventDefault();

        const fullName = document.getElementById('fullName').value;
        const email = document.getElementById('email').value;
        const phone = document.getElementById('phone').value;
        const password = document.getElementById('password').value;
        const confirmPassword = document.getElementById('confirmPassword').value;

        // Validate phone number
        const phoneRegex = /^[0-9]{11}$/;
        if (!phoneRegex.test(phone)) {
            showAlert('Please enter a valid phone number (11 digits)', 'error');
            return;
        }

        if (password !== confirmPassword) {
            showAlert('Passwords do not match!', 'error');
            return;
        }

        const users = JSON.parse(localStorage.getItem('users') || '[]');

        if (users.some(user => user.email === email)) {
            showAlert('Email already registered!', 'error');
            return;
        }

        if (users.some(user => user.phone === phone)) {
            showAlert('Phone number already registered!', 'error');
            return;
        }

        // Create new user object
        const newUser = { fullName, email, phone, password };

        // Add to users array
        users.push(newUser);
        localStorage.setItem('users', JSON.stringify(users));

        // Don't automatically log in - redirect to login page instead
        showAlert('Registration successful! Redirecting to login page...', 'success');
        setTimeout(() => window.location.href = 'login.html', 1500);
    });
}

// Handle Login Form
const loginForm = document.querySelector('.login-form');
if (loginForm) {
    loginForm.addEventListener('submit', (e) => {
        e.preventDefault();

        const email = document.getElementById('email').value;
        const password = document.getElementById('password').value;

        const users = JSON.parse(localStorage.getItem('users') || '[]');
        const user = users.find(u => u.email === email && u.password === password);

        if (user) {
            localStorage.setItem('currentUser', JSON.stringify({
                fullName: user.fullName,
                email: user.email
            }));
            showAlert('Login successful! Redirecting to home page...', 'success');
            setTimeout(() => window.location.href = 'index.html', 1500);
        } else {
            showAlert('Invalid credentials!', 'error');
        }
    });
}

// Logout functionality
const logoutBtn = document.getElementById('logoutBtn');
if (logoutBtn) {
    logoutBtn.addEventListener('click', (e) => {
        e.preventDefault();
        localStorage.removeItem('currentUser');
        // Create and show alert
        const alert = document.createElement('div');
        alert.className = 'alert alert-success fade-in show';
        alert.textContent = 'Logged out successfully!';
        document.body.appendChild(alert);

        // Redirect after delay
        setTimeout(() => {
            window.location.href = 'login.html';
        }, 1000);
    });
}

// Enhanced Alert function with better styling
function showAlert(message, type) {
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} fade-in`;
    alert.textContent = message;

    const form = document.querySelector('form');
    form.parentNode.insertBefore(alert, form);

    // Add fade-in animation
    setTimeout(() => alert.classList.add('show'), 100);

    // Remove alert after delay
    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => alert.remove(), 300);
    }, 3000);
}

// Password visibility toggle
document.addEventListener('DOMContentLoaded', function () {
    const passwordToggles = document.querySelectorAll('.password-toggle');

    passwordToggles.forEach(toggle => {
        toggle.addEventListener('click', function (e) {
            e.preventDefault();

            const input = this.parentElement.querySelector('input');
            const icon = this.querySelector('i');

            // Toggle password visibility
            if (input.type === 'password') {
                input.type = 'text';
                icon.classList.remove('fa-eye');
                icon.classList.add('fa-eye-slash');
                this.classList.add('showing');

                // Auto-hide after 3 seconds
                setTimeout(() => {
                    input.type = 'password';
                    icon.classList.remove('fa-eye-slash');
                    icon.classList.add('fa-eye');
                    this.classList.remove('showing');
                }, 3000);
            } else {
                input.type = 'password';
                icon.classList.remove('fa-eye-slash');
                icon.classList.add('fa-eye');
                this.classList.remove('showing');
            }

            // Add ripple effect
            const ripple = document.createElement('div');
            ripple.classList.add('ripple');
            this.appendChild(ripple);

            setTimeout(() => {
                ripple.remove();
            }, 1000);
        });
    });
});

