import os
import glob
import numpy as np
from PIL import Image, ImageFilter
from agents.image_quality_agent import quality_agent
from agents.diagnosis_agent import diagnosis_agent

def test_agents():
    img_files = glob.glob("Health/eval/images/*.jpg")[:10]
    print(f"Loaded {len(img_files)} test images.\n")

    for idx, f in enumerate(img_files):
        img = Image.open(f)
        q = quality_agent.evaluate(img)
        d = diagnosis_agent.predict(img)
        print(f"[{idx+1}] {os.path.basename(f)}")
        print(f"    Agent 1 (Quality): Status={q.status} | Overall={q.overall_quality}/100")
        print(f"      - Blur Score: {q.blur_score} | Lighting: {q.lighting_score} | Glare: {q.glare_score} | Occlusion: {q.occlusion_score}")
        print(f"      - Triggers: {q.triggered_criteria or 'None'}")
        print(f"    Agent 2 (Diagnosis): Pred={d['predicted_code']} ({d['predicted_class']}) | Conf={d['confidence']*100:.1f}%")
        print(f"      - Certainty: {d['diagnostic_certainty']} | Margin: {d['margin_to_next']:.2f} | Malignancy Risk: {d['cumulative_malignancy_risk']*100:.1f}%")
        print(f"      - Top 3: {[(code, f'{prob*100:.1f}%') for code, name, prob in d['top_3']]}\n")

    # Edge-case testing: Artificially blurred image
    print("=== Testing Synthetic Blur on Image 1 ===")
    sharp_img = Image.open(img_files[0])
    blurry_img = sharp_img.filter(ImageFilter.GaussianBlur(radius=6))
    q_blur = quality_agent.evaluate(blurry_img)
    d_blur = diagnosis_agent.predict(blurry_img)
    print(f"Blurred Image A1 Status: {q_blur.status} (Blur Score: {q_blur.blur_score})")
    print(f"Blurred Image A1 Triggers: {q_blur.triggered_criteria}")
    print(f"Blurred Image A1 Reasoning: {q_blur.reasoning}")
    print(f"Blurred Image A2 Pred: {d_blur['predicted_code']} ({d_blur['confidence']*100:.1f}%)\n")

    # Edge-case testing: Specular glare simulation
    print("=== Testing Synthetic Specular Glare on Image 1 ===")
    glare_arr = np.array(sharp_img).copy()
    glare_arr[50:180, 50:180] = 255  # bright saturated patch
    glare_img = Image.fromarray(glare_arr)
    q_glare = quality_agent.evaluate(glare_img)
    print(f"Glare Image A1 Status: {q_glare.status} (Glare Score: {q_glare.glare_score})")
    print(f"Glare Image A1 Triggers: {q_glare.triggered_criteria}")

    # Test NumPy Fallback on Image 1-5
    print("=== Testing NumPy Fallback Classifier ===")
    for idx in range(min(5, len(img_files))):
        img = Image.open(img_files[idx])
        d_np = diagnosis_agent.predict_numpy_fallback(img)
        print(f"NumPy Fallback Image {idx+1}: {d_np['predicted_code']} ({d_np['confidence']*100:.1f}%), cert={d_np['diagnostic_certainty']}, malig_risk={d_np['cumulative_malignancy_risk']*100:.1f}%")

if __name__ == '__main__':
    test_agents()
