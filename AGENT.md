# AI Assistant Guide: Vehicle Detection and License Plate Recognition

## 1. Project Initialization

### Initial Setup
- Review the README.md for project understanding
- Follow `setup.sh` for environment configuration
- Activate virtual environment and install dependencies

### Model Weights Verification
Ensure model weights are present in:
- Vehicle classification: `workspace/data/runs/results/veh_cls/weights/best.pt`
- License detection: `runs/detect/license3/weights/best.pt`

## 2. Code Structure Overview

### Core Components
- **app.py**: Flask server for routing and UI handling
- **wkg_with_sv.py**: Core detection and tracking logic
- **license.py**: License plate detection and OCR processing
- **Frontend**: `index1.html` and `dashboard1.html`
- **static/**: Resource directory for media and outputs

## 3. Task Workflow

### Issue Management
1. Gather complete context for new tasks/bugs
2. Request clarification when needed
3. Document expected behavior

### Development Process
1. **Local Testing**
   - Write/update unit tests
   - Use sample videos/RTSP streams

2. **Code Modifications**
   - Follow PEP8 style guidelines
   - Maintain clear component separation
   - Ensure UI backward compatibility

3. **Logging**
   - Update JSON logs as needed
   - Verify dashboard endpoint functionality

## 4. Best Practices

### Version Control
- Create feature branches
- Make atomic commits
- Write clear commit messages

### Code Quality
- Implement proper error handling
- Optimize performance
- Maintain security best practices
- Keep documentation updated

## 5. Testing & Validation

### Test Levels
1. **Unit Testing**
   - Focus on critical functions
   - Test core functionality

2. **Integration Testing**
   - Test upload endpoints
   - Verify JSON responses

3. **Manual Testing**
   - UI verification
   - Visual inspection of outputs

## 6. Communication & Updates

### Reporting
- Provide change summaries
- Document testing results
- Create detailed PR descriptions

### User Interaction
- Request review feedback
- Confirm feature completion
- Suggest potential enhancements