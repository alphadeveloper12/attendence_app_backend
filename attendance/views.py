import logging
import base64
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.utils import timezone
from .serializers import EmployeeSerializer
from django.conf import settings
from .models import Employee, Attendance
from .serializers import UserSerializer, AttendanceSerializer
from deepface import DeepFace
from django.core.files.base import ContentFile
import re
from datetime import datetime
from rest_framework.permissions import IsAdminUser
from rest_framework.decorators import permission_classes
from rest_framework.permissions import AllowAny  # This will allow any user to access this view 
from django.contrib.auth import authenticate
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth.models import User
from rest_framework.permissions import IsAdminUser

# Set up logging
logger = logging.getLogger(__name__)

@permission_classes([IsAdminUser])  # Only admin can access this view
class RegisterUserView(APIView):
    def post(self, request):
        try:
            # Retrieve incoming data
            data = request.data
            face_image = data.get('face_image')  # Base64 encoded image
            user_data = data.get('user_data')   # User's info (name, email, etc.)

            logger.info(f"USER_DATA => {user_data}")
            logger.info(f"BASE64_RECEIVED => {face_image[:60]}...")

            if not face_image or not user_data:
                return Response(
                    {"error": "Missing face image or user data"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # -------------------------------
            # 1️⃣ Clean the base64 string
            # -------------------------------
            logger.info("Cleaning base64 image string...")
            base64_pattern = r"^data:image\/(jpeg|png);base64,"
            face_image_data = re.sub(base64_pattern, "", face_image)

            # If the base64 string is still empty, return error
            if not face_image_data:
                return Response({"error": "Invalid base64 image format"}, status=status.HTTP_400_BAD_REQUEST)

            # Decode base64 image
            decoded_image = base64.b64decode(face_image_data)

            # -------------------------------
            # 2️⃣ Create File For Model Saving
            # -------------------------------
            img_file = ContentFile(decoded_image, name="user_face.jpg")
            logger.info(f"IMAGE_SIZE_BYTES => {len(decoded_image)} bytes")

            # -------------------------------
            # 3️⃣ Use DeepFace for Face Detection and Embedding
            # -------------------------------
            logger.info("Running DeepFace for face detection and embedding...")

            # Temporarily save the image for processing
            new_user = Employee(
                name=user_data['name'],
                email=user_data['email'],
                phone=user_data['phone'],
                department=user_data['department'],
                position=user_data['position'],
            )

            # Temporarily save the image to get a file path for DeepFace
            new_user.profile_picture.save("temp_face.jpg", img_file)

            # Path of the saved image for DeepFace to process
            img_path = new_user.profile_picture.path

            # Generate face embeddings using DeepFace
            embeddings = DeepFace.represent(img_path, model_name="VGG-Face")

            # Log the embeddings
            logger.info(f"Face embeddings generated: {embeddings}")

            # Check if no faces were detected
            if not embeddings or len(embeddings) == 0:
                # Remove the temp file & user if no face is detected
                new_user.delete()
                logger.error("No face detected in the image.")
                return Response({"error": "No face detected in image"}, status=400)

            # Check if the embedding is valid
            face_embedding = embeddings[0].get("embedding") if embeddings else None
            if not face_embedding:
                # Remove the temp file & user if face embedding is empty
                new_user.delete()
                logger.error("Face embedding is empty.")
                return Response({"error": "Face embedding could not be generated."}, status=400)

            # -------------------------------
            # 4️⃣ Save Full User Now with face_embedding
            # -------------------------------
            new_user.face_embedding = face_embedding
            new_user.save()  # Only save the user after a valid face is detected and embedded

            logger.info("USER_REGISTERED_SUCCESSFULLY")

            # Build response with the necessary fields
            response_data = {
                "message": "User registered successfully!",
                "user": UserSerializer(new_user).data,
                "image_url": request.build_absolute_uri(new_user.profile_picture.url)
            }

            return Response(response_data, status=status.HTTP_201_CREATED)

        except Exception as e:
            logger.error(f"ERROR => {str(e)}")
            # Return generic error message on failure
            return Response({"error": "An unexpected error occurred."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        
# View to mark attendance for users based on face recognition and selected slot
class MarkAttendanceView(APIView):
    authentication_classes = []  # No authentication needed for this view
    permission_classes = [AllowAny]  # Public access

    def post(self, request):
        try:
            # Retrieve incoming data
            face_image = request.data.get('face_image')  # Base64 encoded image
            slot = request.data.get('slot')  # Slot user selects (e.g., office in, break, break out, office out)

            if not face_image or not slot:
                return Response({"error": "Missing face image or slot selection"}, status=status.HTTP_400_BAD_REQUEST)

            # Clean base64 string (remove unnecessary prefix if any)
            base64_pattern = r"^data:image\/(jpeg|png);base64,"
            face_image_data = re.sub(base64_pattern, "", face_image)

            if not face_image_data:
                return Response({"error": "Invalid base64 image format"}, status=status.HTTP_400_BAD_REQUEST)

            # Decode the image
            decoded_image = base64.b64decode(face_image_data)
            img_file = ContentFile(decoded_image, name="user_face.jpg")

            # Save image temporarily to get a file path for DeepFace
            temp_user = Employee.objects.first()  # Temporary assumption to fetch a user (This can be changed as per requirement)
            temp_user.profile_picture.save("temp_face.jpg", img_file)
            img_path = temp_user.profile_picture.path

            # Generate face embeddings using DeepFace
            embeddings = DeepFace.represent(img_path, model_name="VGG-Face")

            # Check if face embeddings are successfully generated
            if not embeddings or len(embeddings) == 0:
                # Clean up the temporary image after failure
                temp_user.profile_picture.delete()
                return Response({"error": "No face detected in image"}, status=400)

            face_embedding = embeddings[0].get("embedding")

            # Match the face embeddings to a user
            matched_user = None
            for user in Employee.objects.all():
                if user.face_embedding and self.compare_embeddings(user.face_embedding, face_embedding):
                    matched_user = user
                    break

            if not matched_user:
                # Clean up the temporary image after failure
                temp_user.profile_picture.delete()
                return Response({"error": "User not found or face not recognized"}, status=400)

            # Mark attendance for the selected slot
            print("Matched User:", matched_user.name)
            
            # Get current date and time
            now = timezone.localtime()

            today = now.date()
            
            # Get or create attendance record for today
            # Using filter().first() to handle multiple records gracefully
            attendance = Attendance.objects.filter(
                user=matched_user,
                date=today
            ).first()
            
            if not attendance:
                # Create new attendance record if none exists
                attendance = Attendance.objects.create(
                    user=matched_user,
                    date=today,
                    status='present'
                )
            else:
                # Update existing record
                attendance.status = 'present'

            # Handle the different slot types - use actual current time
            if slot == 'office_in':
                attendance.check_in_time = now
            elif slot == 'break_in':
                attendance.break_in_time = now
            elif slot == 'break_out':
                attendance.break_out_time = now
            elif slot == 'office_out':
                attendance.check_out_time = now

            # Calculate late or early minutes based on time differences
            attendance.calculate_late_and_early()

            # Clean up the temporary image after success
            temp_user.profile_picture.delete()

            # Prepare response time based on slot
            response_time = None
            if slot == 'office_in' and attendance.check_in_time:
                response_time = attendance.check_in_time.strftime('%I:%M %p')
            elif slot == 'break_in' and attendance.break_in_time:
                response_time = attendance.break_in_time.strftime('%I:%M %p')
            elif slot == 'break_out' and attendance.break_out_time:
                response_time = attendance.break_out_time.strftime('%I:%M %p')
            elif slot == 'office_out' and attendance.check_out_time:
                response_time = attendance.check_out_time.strftime('%I:%M %p')

            return Response({
                "message": f"Attendance marked successfully for {slot}.",
                "user": matched_user.name,
                "slot": slot,
                "time": response_time,
                "late_minutes": attendance.late_minutes,
                "early_minutes": attendance.early_minutes
            }, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"ERROR => {str(e)}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def compare_embeddings(self, stored_embedding, new_embedding, threshold=0.5):
        """
        Compare two face embeddings and return True if they are similar within a threshold.
        """
        from scipy.spatial.distance import cosine
        # Compute cosine distance between two embeddings, lower values mean closer embeddings
        distance = cosine(stored_embedding, new_embedding)
        return distance < threshold  # You can adjust the threshold as needed
    


class AdminLoginView(APIView):
    """
    Admin Login using email and password to get JWT token.
    """
    permission_classes = [AllowAny]  # Allow any user to access this view (no authentication required)

    def post(self, request):
        # Retrieve email and password from the request body
        email = request.data.get("email")
        password = request.data.get("password")
        print (email, password)

        # Validate the data
        if not email or not password:
            return Response(
                {"error": "Email and password are required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Authenticate the user
        try:
            # Look up the user by email
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            return Response(
                {"error": "Admin with this email does not exist."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Verify if the user is an admin
        if not user.is_superuser:
            return Response(
                {"error": "You must be an admin to login."},
                status=status.HTTP_403_FORBIDDEN,
            )

        # Authenticate the user using the email and password
        user = authenticate(request, username=user.username, password=password)

        if user is not None:
            # Create a JWT token for the authenticated admin user
            refresh = RefreshToken.for_user(user)
            access_token = refresh.access_token

            return Response(
                {"message": "Login successful", "access_token": str(access_token)},
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                {"error": "Invalid email or password"},
                status=status.HTTP_400_BAD_REQUEST,
            )

@permission_classes([IsAdminUser])  # Only admin can access this view
class AttendanceStatsView(APIView):
    """
    View to get the total number of employees and today's attendance count.
    """
    def get(self, request):
        try:
            # Get total number of employees
            total_employees = Employee.objects.count()

            # Get today's date
            today = timezone.localdate()

            # Get today's attendance count
            today_attendance_count = Attendance.objects.filter(date=today).count()

            # Return the response
            return Response({
                "total_employees": total_employees,
                "today_attendance_count": today_attendance_count
            }, status=status.HTTP_200_OK)

        except Exception as e:
            # If there is an error, return an internal server error
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        
        
@permission_classes([IsAdminUser])  # Only admin can access this view
class EmployeeListView(APIView):
 # Optional: if you want to restrict this API to authenticated users

    def get(self, request):
        # Fetch all employees
        employees = Employee.objects.all()

        # Serialize the employees' data
        serializer = EmployeeSerializer(employees, many=True, context={'request': request})

        # Return the data in the response
        return Response(serializer.data)