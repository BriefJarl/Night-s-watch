from app.core.database import SessionLocal
from app.schemas.camera import CameraCreate, CameraUpdate
from app.services.camera_service import CameraService


def test_camera_service():

    db = SessionLocal()

    service = CameraService()

    try:

        # -------------------------
        # CREATE CAMERA
        # -------------------------

        camera_data = CameraCreate(
            name="Main Gate Camera",
            location="Main Entrance",
            stream_url="0",
            is_active=True,
        )

        camera = service.create_camera(
            db=db,
            camera_data=camera_data,
        )

        print("Camera created successfully!")
        print("Camera ID:", camera.id)


        # -------------------------
        # GET CAMERA
        # -------------------------

        fetched_camera = service.get_camera(
            db=db,
            camera_id=camera.id,
        )

        if fetched_camera is None:

            raise Exception(
                "Camera could not be fetched"
            )

        print("Camera fetched successfully!")
        print("Camera Name:", fetched_camera.name)


        # -------------------------
        # UPDATE CAMERA
        # -------------------------

        update_data = CameraUpdate(
            location="Updated Main Entrance"
        )

        updated_camera = service.update_camera(
            db=db,
            camera_id=camera.id,
            camera_data=update_data,
        )

        if updated_camera is None:

            raise Exception(
                "Camera could not be updated"
            )

        print("Camera updated successfully!")
        print(
            "Updated Location:",
            updated_camera.location,
        )


        # -------------------------
        # GET ALL CAMERAS
        # -------------------------

        cameras = service.get_cameras(
            db=db,
            skip=0,
            limit=100,
        )

        print(
            "Total Cameras:",
            len(cameras),
        )


        # -------------------------
        # DELETE CAMERA
        # -------------------------

        deleted = service.delete_camera(
            db=db,
            camera_id=camera.id,
        )

        if not deleted:

            raise Exception(
                "Camera could not be deleted"
            )

        print("Camera deleted successfully!")

        print()
        print(
            "Camera service tests successful!"
        )


    finally:

        db.close()


if __name__ == "__main__":

    test_camera_service()