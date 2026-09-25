# -*- coding: utf-8 -*-
# Dr. Florent Poux, 3D Geodata Academy
# https://learngeodata.eu
#
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Florent Poux
#
# Part of https://github.com/florentPoux/3d-data-materials
# Article:  https://medium.com/data-science-collective/how-to-build-a-python-gui-for-3d-scene-labeling-49dd43624a7f
# Dataset:  https://learngeodata.eu/materials/3d-scene-labeling-gui/
#
# Author of 3D Data Science with Python (O'Reilly Media, 2025).
# ORCID 0000-0001-6368-4399
"""
Interactive Painting Tools for Multi-Class Image Masking
=========================================================

This module provides interactive tools for painting multi-class masks
on images using OpenCV, designed for 3D semantic labeling workflows.

Tools:
- HDMultiClassMaskPainter: HD interactive masking with large display
- paint_mask_hd: Launch HD masking on a single image
- mask_multiple_images_hd: Batch HD masking across multiple images

Author: Florent Poux
License: MIT (see the SPDX line above)
"""

import numpy as np
import cv2

#%% HD Masking Functions with Large Display

class HDMultiClassMaskPainter:
    """Interactive HD masking tool with large display for better visibility"""

    def __init__(self, image, num_classes=5, target_height=1080):
        # Resize image to HD for better visibility
        self.original_image = image.copy()
        self.original_shape = image.shape[:2]

        # Calculate scale to reach target height while maintaining aspect ratio
        scale = target_height / image.shape[0]
        new_width = int(image.shape[1] * scale)
        new_height = target_height

        self.image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        self.display_shape = self.image.shape[:2]
        self.scale_factor = scale

        # Initialize mask at display resolution
        self.mask = np.zeros(self.display_shape, dtype=np.uint8)
        self.current_class = 1
        self.brush_size = 30
        self.drawing = False
        self.num_classes = num_classes

        # Color palette
        self.class_colors = [
            (0, 0, 0),       # 0: Background
            (255, 0, 0),     # 1: Red
            (0, 255, 0),     # 2: Green
            (0, 0, 255),     # 3: Blue
            (255, 255, 0),   # 4: Yellow
            (255, 0, 255),   # 5: Magenta
        ]

        print(f"HD Masking Tool: Displaying at {new_width}x{new_height} (scale: {scale:.2f}x)")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.paint_mask(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.paint_mask(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False

    def paint_mask(self, x, y):
        cv2.circle(self.mask, (x, y), self.brush_size, self.current_class, -1)

    def get_overlay(self):
        overlay = self.image.copy()
        for class_id in range(1, self.num_classes + 1):
            mask_class = (self.mask == class_id)
            overlay[mask_class] = (0.6 * self.image[mask_class] +
                                   0.4 * np.array(self.class_colors[class_id]))
        return overlay.astype(np.uint8)

    def get_original_resolution_mask(self):
        """Resize mask back to original image resolution"""
        mask_original = cv2.resize(self.mask,
                                   (self.original_shape[1], self.original_shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        return mask_original

    def run(self, window_name="HD Mask Painter"):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        print("=== HD Mask Painting Tool ===")
        print("Keys: 1-5 = Select class | +/- = Brush size | 's' = Save | 'c' = Clear | 'q' = Quit")

        while True:
            overlay = self.get_overlay()

            # Add UI info with better visibility
            info_text = f"Class: {self.current_class} | Brush: {self.brush_size}"
            cv2.putText(overlay, info_text, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(overlay, info_text, (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

            # Show class legend
            legend_y = 100
            for i in range(1, self.num_classes + 1):
                color = self.class_colors[i]
                cv2.rectangle(overlay, (20, legend_y), (60, legend_y + 30), color, -1)
                cv2.rectangle(overlay, (20, legend_y), (60, legend_y + 30), (255, 255, 255), 2)
                legend_text = f"Class {i}"
                cv2.putText(overlay, legend_text, (70, legend_y + 22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                legend_y += 40

            cv2.imshow(window_name, overlay)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.destroyAllWindows()
                return self.get_original_resolution_mask()
            elif key == ord('c'):
                self.mask = np.zeros(self.display_shape, dtype=np.uint8)
                print("Mask cleared")
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                self.current_class = int(chr(key))
                print(f"Selected class: {self.current_class}")
            elif key == ord('+') or key == ord('='):
                self.brush_size = min(100, self.brush_size + 10)
                print(f"Brush size: {self.brush_size}")
            elif key == ord('-') or key == ord('_'):
                self.brush_size = max(10, self.brush_size - 10)
                print(f"Brush size: {self.brush_size}")

        cv2.destroyAllWindows()
        return self.get_original_resolution_mask()

def paint_mask_hd(image, num_classes=5, target_height=1080):
    """Launch HD masking tool with large display for better visibility"""
    painter = HDMultiClassMaskPainter(image, num_classes, target_height)
    mask = painter.run()
    return mask

def mask_multiple_images_hd(images, num_images=5, num_classes=5, target_height=1080):
    """Interactively mask multiple images in HD with large display"""
    masks = []
    num_to_mask = min(num_images, len(images))

    print(f"\n=== HD Masking for {num_to_mask} Images ===")

    for i in range(num_to_mask):
        print(f"\nMasking image {i+1}/{num_to_mask} in HD")
        mask = paint_mask_hd(images[i], num_classes, target_height)
        masks.append(mask)
        print(f"  Mask {i+1} completed with {len(np.unique(mask))-1} classes")

    return masks
