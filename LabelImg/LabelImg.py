import cv2
import os

# === Paths ===
#add your img folder address here
IMAGE_FOLDER = r"C:\Users\PC\Downloads\periodogram"
LABEL_FOLDER = r"C:\Users\PC\Downloads\labels"

# === Globals ===
zoom_box = []
zoom_factor = 5
image_list = []
image_index = 0
original_image = None
display_image = None
current_image_path = ""
stage = 0  # 0 = selecting zoom, 1 = labeling
clicks = []
annotations = []

def reset_state():
    global clicks, zoom_box, display_image, stage
    clicks = []
    zoom_box = []
    display_image = None
    stage = 0

def save_labels(annotations, shape, filename):
    h_img, w_img = shape
    label_path = os.path.join(LABEL_FOLDER, os.path.splitext(os.path.basename(filename))[0] + ".txt")
    with open(label_path, "w") as f:
        for class_id, ann in enumerate(annotations):
            cx, cy = ann[0]
            ex, ey = ann[1]
            box_w = abs(ex - cx) * 2
            box_h = abs(ey - cy) * 2
            x_center = cx / w_img
            y_center = cy / h_img
            w_norm = box_w / w_img
            h_norm = box_h / h_img
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")


def mouse_callback(event, x, y, flags, param):
    global clicks, zoom_box, display_image, stage, annotations, original_image

    if event == cv2.EVENT_LBUTTONDOWN:
        if stage == 0:
            zoom_box.append((x, y))
            if len(zoom_box) == 2:
                x1, y1 = zoom_box[0]
                x2, y2 = zoom_box[1]
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)
                roi = original_image[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    print(" Invalid zoom area.")
                    zoom_box.clear()
                    return
                display_image = cv2.resize(roi, (roi.shape[1]*zoom_factor, roi.shape[0]*zoom_factor), interpolation=cv2.INTER_LINEAR)
                stage = 1
                clicks.clear()
                cv2.imshow("Labeling", display_image)

        elif stage == 1:
            clicks.append((x, y))
            if len(clicks) == 2:
                # Convert from zoomed clicks to original image coords
                (zx1, zy1), (zx2, zy2) = zoom_box
                x_zoom_min, x_zoom_max = min(zx1, zx2), max(zx1, zx2)
                y_zoom_min, y_zoom_max = min(zy1, zy2), max(zy1, zy2)

                center_x = clicks[0][0] / zoom_factor + x_zoom_min
                center_y = clicks[0][1] / zoom_factor + y_zoom_min
                edge_x = clicks[1][0] / zoom_factor + x_zoom_min
                edge_y = clicks[1][1] / zoom_factor + y_zoom_min

                annotations.append(((center_x, center_y), (edge_x, edge_y)))

                # Draw the box
                box_w = abs(edge_x - center_x) * 2
                box_h = abs(edge_y - center_y) * 2
                top_left = (int(center_x - box_w / 2), int(center_y - box_h / 2))
                bottom_right = (int(center_x + box_w / 2), int(center_y + box_h / 2))
                cv2.rectangle(original_image, top_left, bottom_right, (0, 255, 0), 2)

                print(f" Added box #{len(annotations)}")
                # Reset
                zoom_box.clear()
                clicks.clear()
                stage = 0
                cv2.imshow("Labeling", original_image)

def load_image(index):
    global original_image, current_image_path
    current_image_path = image_list[index]
    original_image = cv2.imread(current_image_path)
    if original_image is None:
        print(f" Failed to load: {current_image_path}")
        return False
    cv2.imshow("Labeling", original_image)
    cv2.setWindowTitle("Labeling", f"{os.path.basename(current_image_path)} ({index + 1}/{len(image_list)})")
    return True

def main():
    global image_index, image_list, annotations

    os.makedirs(LABEL_FOLDER, exist_ok=True)
    image_list.extend(sorted([
        os.path.join(IMAGE_FOLDER, f)
        for f in os.listdir(IMAGE_FOLDER)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]))

    if not image_list:
        print("No images found.")
        return

    cv2.namedWindow("Labeling")
    cv2.setMouseCallback("Labeling", mouse_callback)

    while image_index < len(image_list):
        reset_state()
        annotations = []

        if not load_image(image_index):
            image_index += 1
            continue

        while True:
            key = cv2.waitKey(0)

            if key == 27:  # ESC
                print(" Exiting.")
                cv2.destroyAllWindows()
                return

            elif key in [13, ord('n')]:  # Enter or 'n'
                if annotations:
                    save_labels(annotations, original_image.shape[:2], current_image_path)
                    print(f"[{image_index+1}/{len(image_list)}]  Saved {len(annotations)} boxes")
                else:
                    print(" No labels to save.")
                image_index += 1
                break

            elif key == ord('r'):
                print(" Resetting current image...")
                annotations.clear()
                reset_state()
                load_image(image_index)

    print(" Done labeling.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
