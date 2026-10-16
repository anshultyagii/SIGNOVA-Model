# SignSpeakAI — Independent Recorded Samples Evaluation Report
**Generated at:** 2026-10-14 15:51:05

- **Gallery Directory:** `F:\SignSpeakAI\dataset\processed_corrected`
- **Evaluation Dataset:** `F:\SignSpeakAI\dataset\recorded_processed`
- **Total Files Found:** 84
- **Evaluated Samples:** 84
- **Skipped Samples:** 0

## 1. Overall Executive Summary
| Metric | Count / Score | Percentage |
| :--- | :---: | :---: |
| **Top-1 Accuracy** | 3 / 84 | **3.57%** |
| **Top-3 Accuracy** | 15 / 84 | **17.86%** |
| **Top-1 Error Count** | 81 / 84 | 96.43% |
| **Evaluated Classes** | 25 | 100% Gallery Coverage |

### 1.1 Data Quality & Hand Detection Audit
| Cohort | Sample Count | Top-1 Accuracy | Top-3 Accuracy | Characteristic |
| :--- | :---: | :---: | :---: | :--- |
| **Samples with Detected Hands** | 12 / 84 (14.3%) | **16.67%** | **25.00%** | Hand landmarks present in >= 1 frame |
| **Samples with Zero Hand Detections** | 72 / 84 (85.7%) | **1.39%** | **16.67%** | MediaPipe hand presence completely absent (0 frames) |

> [!NOTE]
> **Root Cause Analysis for Low Baseline Accuracy:**
> In **72 of 84** (85.7%) recorded samples, MediaPipe extracted zero hand landmarks (`presence[:, 2:] == False` across all frames).
> Without hand landmarks, hand shape (`w_shape=0.50`) and hand position (`w_pos=0.25`) cannot be matched, triggering missing-hand penalties and forcing recognition to rely solely on coarse torso pose (`w_pose=0.25`). This caused severe prediction collapse into shorter templates (`Give` and `What`).

## 2. Per-Class Accuracy Breakdown
| Class | Total | Top-1 Correct | Top-1 Acc | Top-3 Correct | Top-3 Acc | Mean Sim | Mean Dist |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Drink** | 10 | 1 | 10.0% | 9 | 90.0% | 0.582 | 0.725 |
| **Eat** | 10 | 0 | 0.0% | 0 | 0.0% | 0.542 | 0.876 |
| **Eight** | 1 | 0 | 0.0% | 0 | 0.0% | 0.593 | 0.687 |
| **Five** | 1 | 0 | 0.0% | 0 | 0.0% | 0.585 | 0.708 |
| **Food** | 1 | 0 | 0.0% | 0 | 0.0% | 0.598 | 0.673 |
| **Four** | 1 | 0 | 0.0% | 0 | 0.0% | 0.593 | 0.687 |
| **Give** | 1 | 0 | 0.0% | 1 | 100.0% | 0.587 | 0.705 |
| **Go** | 1 | 0 | 0.0% | 0 | 0.0% | 0.665 | 0.504 |
| **Help** | 10 | 0 | 0.0% | 0 | 0.0% | 0.571 | 0.796 |
| **I_Me** | 10 | 1 | 10.0% | 2 | 20.0% | 0.565 | 0.774 |
| **Milk** | 1 | 0 | 0.0% | 0 | 0.0% | 0.724 | 0.382 |
| **My** | 1 | 0 | 0.0% | 0 | 0.0% | 0.586 | 0.708 |
| **Nine** | 1 | 0 | 0.0% | 0 | 0.0% | 0.591 | 0.693 |
| **One** | 10 | 0 | 0.0% | 2 | 20.0% | 0.577 | 0.732 |
| **Seven** | 1 | 0 | 0.0% | 0 | 0.0% | 0.589 | 0.698 |
| **Six** | 1 | 0 | 0.0% | 0 | 0.0% | 0.693 | 0.444 |
| **Ten** | 1 | 0 | 0.0% | 0 | 0.0% | 0.590 | 0.695 |
| **Three** | 1 | 0 | 0.0% | 0 | 0.0% | 0.597 | 0.675 |
| **Two** | 6 | 0 | 0.0% | 0 | 0.0% | 0.563 | 0.785 |
| **Want_Desire_Wish_For** | 1 | 0 | 0.0% | 0 | 0.0% | 0.587 | 0.704 |
| **Water** | 10 | 0 | 0.0% | 0 | 0.0% | 0.570 | 0.755 |
| **We** | 1 | 0 | 0.0% | 0 | 0.0% | 0.586 | 0.707 |
| **What** | 1 | 1 | 100.0% | 1 | 100.0% | 0.585 | 0.711 |
| **Where** | 1 | 0 | 0.0% | 0 | 0.0% | 0.744 | 0.344 |
| **You** | 1 | 0 | 0.0% | 0 | 0.0% | 0.585 | 0.709 |

## 3. Misclassifications Analysis
A total of 81 samples had Top-1 prediction mismatches:

| Sample Name | Ground Truth | Top-1 Predicted | Sim | Dist | True Rank | Top-3 Candidates |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `Drink_sample_002_landmarks.npz` | **Drink** | `Give` | 0.574 | 0.743 | #3 | Give(0.57), What(0.57), Drink(0.57) |
| `Drink_sample_003_landmarks.npz` | **Drink** | `Give` | 0.568 | 0.761 | #2 | Give(0.57), Drink(0.57), What(0.57) |
| `Drink_sample_004_landmarks.npz` | **Drink** | `Give` | 0.561 | 0.783 | #3 | Give(0.56), What(0.56), Drink(0.56) |
| `Drink_sample_005_landmarks.npz` | **Drink** | `Give` | 0.570 | 0.756 | #2 | Give(0.57), Drink(0.56), What(0.56) |
| `Drink_sample_006_landmarks.npz` | **Drink** | `Give` | 0.565 | 0.771 | #2 | Give(0.56), Drink(0.56), What(0.56) |
| `Drink_sample_007_landmarks.npz` | **Drink** | `What` | 0.575 | 0.741 | #3 | What(0.57), Give(0.57), Drink(0.57) |
| `Drink_sample_008_landmarks.npz` | **Drink** | `What` | 0.576 | 0.737 | #4 | What(0.58), Give(0.57), I_Me(0.57) |
| `Drink_sample_009_landmarks.npz` | **Drink** | `Give` | 0.569 | 0.757 | #3 | Give(0.57), What(0.56), Drink(0.56) |
| `Drink_sample_010_landmarks.npz` | **Drink** | `Give` | 0.564 | 0.773 | #2 | Give(0.56), Drink(0.56), What(0.56) |
| `Eat_sample_001_landmarks.npz` | **Eat** | `Give` | 0.592 | 0.690 | #10 | Give(0.59), What(0.59), 1_One(0.58) |
| `Eat_sample_002_landmarks.npz` | **Eat** | `Give` | 0.562 | 0.778 | #11 | Give(0.56), Drink(0.56), What(0.56) |
| `Eat_sample_003_landmarks.npz` | **Eat** | `Give` | 0.558 | 0.792 | #11 | Give(0.56), Drink(0.56), What(0.55) |
| `Eat_sample_004_landmarks.npz` | **Eat** | `What` | 0.567 | 0.764 | #11 | What(0.57), Give(0.57), I_Me(0.56) |
| `Eat_sample_005_landmarks.npz` | **Eat** | `Give` | 0.557 | 0.795 | #11 | Give(0.56), Drink(0.55), What(0.55) |
| `Eat_sample_006_landmarks.npz` | **Eat** | `Give` | 0.557 | 0.795 | #11 | Give(0.56), Drink(0.56), What(0.55) |
| `Eat_sample_007_landmarks.npz` | **Eat** | `Give` | 0.369 | 1.711 | #6 | Give(0.37), Drink(0.35), I_Me(0.35) |
| `Eat_sample_008_landmarks.npz` | **Eat** | `Give` | 0.549 | 0.821 | #11 | Give(0.55), What(0.55), Drink(0.54) |
| `Eat_sample_009_landmarks.npz` | **Eat** | `Give` | 0.553 | 0.808 | #11 | Give(0.55), What(0.55), Drink(0.55) |
| `Eat_sample_010_landmarks.npz` | **Eat** | `Give` | 0.553 | 0.807 | #11 | Give(0.55), Drink(0.55), What(0.55) |
| `Eight_sample_001_landmarks.npz` | **Eight** | `Give` | 0.593 | 0.687 | #18 | Give(0.59), What(0.59), Drink(0.58) |
| `Five_sample_001_landmarks.npz` | **Five** | `What` | 0.585 | 0.708 | #17 | What(0.59), Give(0.58), 1_One(0.58) |
| `Food_sample_001_landmarks.npz` | **Food** | `Give` | 0.598 | 0.673 | #15 | Give(0.60), What(0.60), 1_One(0.59) |
| `Four_sample_001_landmarks.npz` | **Four** | `Give` | 0.593 | 0.687 | #20 | Give(0.59), What(0.59), 1_One(0.58) |
| `Give_sample_001_landmarks.npz` | **Give** | `What` | 0.587 | 0.705 | #2 | What(0.59), Give(0.58), 1_One(0.58) |
| `Go_sample_001_landmarks.npz` | **Go** | `One` | 0.665 | 0.504 | #16 | 1_One(0.66), 3_Three(0.65), 2_Two(0.65) |
| `Help_sample_001_landmarks.npz` | **Help** | `Six` | 0.699 | 0.431 | #23 | 6_Six(0.70), Water(0.68), 7_Seven(0.68) |
| `Help_sample_002_landmarks.npz` | **Help** | `What` | 0.570 | 0.755 | #23 | What(0.57), Give(0.57), Drink(0.56) |
| `Help_sample_003_landmarks.npz` | **Help** | `What` | 0.568 | 0.762 | #23 | What(0.57), Give(0.57), I_Me(0.56) |
| `Help_sample_004_landmarks.npz` | **Help** | `My` | 0.679 | 0.473 | #21 | My(0.68), Food(0.67), 7_Seven(0.67) |
| `Help_sample_005_landmarks.npz` | **Help** | `Give` | 0.565 | 0.771 | #23 | Give(0.56), Drink(0.56), What(0.56) |
| `Help_sample_006_landmarks.npz` | **Help** | `Drink` | 0.562 | 0.780 | #23 | Drink(0.56), Give(0.56), What(0.56) |
| `Help_sample_007_landmarks.npz` | **Help** | `Give` | 0.563 | 0.777 | #23 | Give(0.56), Drink(0.56), What(0.56) |
| `Help_sample_008_landmarks.npz` | **Help** | `Give` | 0.565 | 0.770 | #23 | Give(0.56), Drink(0.56), What(0.56) |
| `Help_sample_009_landmarks.npz` | **Help** | `Give` | 0.376 | 1.663 | #24 | Give(0.38), Drink(0.36), I_Me(0.35) |
| `Help_sample_010_landmarks.npz` | **Help** | `Drink` | 0.562 | 0.780 | #23 | Drink(0.56), Give(0.56), What(0.56) |
| `I_Me_sample_001_landmarks.npz` | **I_Me** | `Give` | 0.597 | 0.675 | #4 | Give(0.60), What(0.59), 1_One(0.59) |
| `I_Me_sample_002_landmarks.npz` | **I_Me** | `Drink` | 0.569 | 0.757 | #4 | Drink(0.57), Give(0.57), What(0.56) |
| `I_Me_sample_003_landmarks.npz` | **I_Me** | `Drink` | 0.569 | 0.757 | #4 | Drink(0.57), Give(0.57), What(0.56) |
| `I_Me_sample_004_landmarks.npz` | **I_Me** | `Drink` | 0.573 | 0.745 | #4 | Drink(0.57), Give(0.57), What(0.56) |
| `I_Me_sample_006_landmarks.npz` | **I_Me** | `Drink` | 0.563 | 0.775 | #4 | Drink(0.56), Give(0.56), What(0.56) |
| `I_Me_sample_007_landmarks.npz` | **I_Me** | `Drink` | 0.565 | 0.769 | #4 | Drink(0.57), Give(0.56), What(0.56) |
| `I_Me_sample_008_landmarks.npz` | **I_Me** | `Drink` | 0.565 | 0.771 | #4 | Drink(0.56), Give(0.56), What(0.56) |
| `I_Me_sample_009_landmarks.npz` | **I_Me** | `One` | 0.488 | 1.048 | #2 | 1_One(0.49), I_Me(0.49), You(0.48) |
| `I_Me_sample_010_landmarks.npz` | **I_Me** | `Drink` | 0.564 | 0.775 | #4 | Drink(0.56), Give(0.56), What(0.56) |
| `Milk_sample_001_landmarks.npz` | **Milk** | `Food` | 0.724 | 0.382 | #25 | Food(0.72), 2_Two(0.69), 3_Three(0.69) |
| `My_sample_001_landmarks.npz` | **My** | `Give` | 0.586 | 0.708 | #8 | Give(0.59), What(0.58), Drink(0.58) |
| `Nine_sample_001_landmarks.npz` | **Nine** | `Give` | 0.591 | 0.693 | #19 | Give(0.59), What(0.59), 1_One(0.58) |
| `One_sample_001_landmarks.npz` | **One** | `What` | 0.588 | 0.700 | #3 | What(0.59), Give(0.59), 1_One(0.58) |
| `One_sample_002_landmarks.npz` | **One** | `What` | 0.577 | 0.733 | #4 | What(0.58), Give(0.58), I_Me(0.57) |
| `One_sample_003_landmarks.npz` | **One** | `What` | 0.576 | 0.736 | #4 | What(0.58), Give(0.58), I_Me(0.57) |
| `One_sample_004_landmarks.npz` | **One** | `What` | 0.578 | 0.731 | #4 | What(0.58), Give(0.58), I_Me(0.57) |
| `One_sample_005_landmarks.npz` | **One** | `What` | 0.577 | 0.734 | #4 | What(0.58), Give(0.58), I_Me(0.57) |
| `One_sample_006_landmarks.npz` | **One** | `What` | 0.577 | 0.734 | #4 | What(0.58), Give(0.58), I_Me(0.57) |
| `One_sample_007_landmarks.npz` | **One** | `What` | 0.571 | 0.753 | #4 | What(0.57), Give(0.57), I_Me(0.56) |
| `One_sample_008_landmarks.npz` | **One** | `What` | 0.579 | 0.728 | #4 | What(0.58), Give(0.58), I_Me(0.57) |
| `One_sample_009_landmarks.npz` | **One** | `What` | 0.579 | 0.728 | #4 | What(0.58), Give(0.58), I_Me(0.57) |
| `One_sample_010_landmarks.npz` | **One** | `What` | 0.574 | 0.743 | #3 | What(0.57), Give(0.57), 1_One(0.56) |
| `Seven_sample_001_landmarks.npz` | **Seven** | `What` | 0.589 | 0.698 | #15 | What(0.59), Give(0.59), 1_One(0.58) |
| `Six_sample_001_landmarks.npz` | **Six** | `Food` | 0.693 | 0.444 | #10 | Food(0.69), 4_Four(0.66), 3_Three(0.66) |
| `Ten_sample_001_landmarks.npz` | **Ten** | `Give` | 0.590 | 0.695 | #16 | Give(0.59), What(0.59), 1_One(0.58) |
| `Three_sample_001_landmarks.npz` | **Three** | `Give` | 0.597 | 0.675 | #11 | Give(0.60), What(0.60), 1_One(0.59) |
| `Two_sample_001_landmarks.npz` | **Two** | `We` | 0.487 | 1.052 | #17 | We(0.49), Want_Desire_Wish_For(0.40), Milk(0.40) |
| `Two_sample_002_landmarks.npz` | **Two** | `What` | 0.578 | 0.729 | #7 | What(0.58), Give(0.58), 1_One(0.57) |
| `Two_sample_003_landmarks.npz` | **Two** | `What` | 0.575 | 0.738 | #7 | What(0.58), Give(0.57), I_Me(0.57) |
| `Two_sample_004_landmarks.npz` | **Two** | `What` | 0.581 | 0.722 | #6 | What(0.58), Give(0.58), 1_One(0.57) |
| `Two_sample_005_landmarks.npz` | **Two** | `What` | 0.578 | 0.730 | #7 | What(0.58), Give(0.58), 1_One(0.57) |
| `Two_sample_006_landmarks.npz` | **Two** | `What` | 0.576 | 0.737 | #8 | What(0.58), Give(0.57), 1_One(0.57) |
| `Want_Desire_Wish_For_sample_001_landmarks.npz` | **Want_Desire_Wish_For** | `What` | 0.587 | 0.704 | #21 | What(0.59), Give(0.58), 1_One(0.57) |
| `Water_sample_001_landmarks.npz` | **Water** | `What` | 0.585 | 0.710 | #11 | What(0.58), Give(0.58), 1_One(0.57) |
| `Water_sample_002_landmarks.npz` | **Water** | `Give` | 0.566 | 0.767 | #10 | Give(0.57), Drink(0.56), What(0.56) |
| `Water_sample_003_landmarks.npz` | **Water** | `Drink` | 0.565 | 0.769 | #10 | Drink(0.57), Give(0.56), What(0.56) |
| `Water_sample_004_landmarks.npz` | **Water** | `Drink` | 0.570 | 0.755 | #10 | Drink(0.57), Give(0.57), What(0.56) |
| `Water_sample_005_landmarks.npz` | **Water** | `Drink` | 0.566 | 0.766 | #10 | Drink(0.57), Give(0.56), What(0.56) |
| `Water_sample_006_landmarks.npz` | **Water** | `Drink` | 0.570 | 0.754 | #10 | Drink(0.57), Give(0.57), What(0.56) |
| `Water_sample_007_landmarks.npz` | **Water** | `Give` | 0.567 | 0.764 | #10 | Give(0.57), Drink(0.57), What(0.56) |
| `Water_sample_008_landmarks.npz` | **Water** | `Drink` | 0.571 | 0.752 | #10 | Drink(0.57), Give(0.57), What(0.57) |
| `Water_sample_009_landmarks.npz` | **Water** | `Give` | 0.571 | 0.751 | #10 | Give(0.57), Drink(0.57), What(0.57) |
| `Water_sample_010_landmarks.npz` | **Water** | `Drink` | 0.568 | 0.761 | #10 | Drink(0.57), Give(0.57), What(0.56) |
| `We_sample_001_landmarks.npz` | **We** | `What` | 0.586 | 0.707 | #21 | What(0.59), Give(0.58), 1_One(0.58) |
| `Where_sample_001_landmarks.npz` | **Where** | `Food` | 0.744 | 0.344 | #23 | Food(0.74), 3_Three(0.69), 2_Two(0.68) |
| `You_sample_001_landmarks.npz` | **You** | `What` | 0.585 | 0.709 | #9 | What(0.59), Give(0.58), 1_One(0.57) |

## 4. Skipped Samples
No samples were skipped. All samples were successfully parsed, normalized, and evaluated.

## 5. Confusion Matrix (Row = True, Col = Predicted)
```text
True \ Pred | Drin  Eat Eigh Five Food Four Give   Go Help I_Me Milk   My Nine  One Seve  Six  Ten Thre  Two Want Wate   We What Wher  You
------------------------------------------------------------------------------------------------------------------------------------------
Drink       |    1    .    .    .    .    .    7    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    2    .    .
Eat         |    .    .    .    .    .    .    9    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    1    .    .
Eight       |    .    .    .    .    .    .    1    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .
Five        |    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    1    .    .
Food        |    .    .    .    .    .    .    1    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .
Four        |    .    .    .    .    .    .    1    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .
Give        |    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    1    .    .
Go          |    .    .    .    .    .    .    .    .    .    .    .    .    .    1    .    .    .    .    .    .    .    .    .    .    .
Help        |    2    .    .    .    .    .    4    .    .    .    .    1    .    .    .    1    .    .    .    .    .    .    2    .    .
I_Me        |    7    .    .    .    .    .    1    .    .    1    .    .    .    1    .    .    .    .    .    .    .    .    .    .    .
Milk        |    .    .    .    .    1    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .
My          |    .    .    .    .    .    .    1    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .
Nine        |    .    .    .    .    .    .    1    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .
One         |    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .   10    .    .
Seven       |    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    1    .    .
Six         |    .    .    .    .    1    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .
Ten         |    .    .    .    .    .    .    1    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .
Three       |    .    .    .    .    .    .    1    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .
Two         |    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    1    5    .    .
Want_Desire |    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    1    .    .
Water       |    6    .    .    .    .    .    3    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    1    .    .
We          |    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    1    .    .
What        |    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    1    .    .
Where       |    .    .    .    .    1    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .
You         |    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    .    1    .    .
```

## 6. Known Limitations & Technical Observations
1. **Single-Exemplar Representation:** The enrolled gallery uses exactly one exemplar video per sign. Variations in signing style, speed, and morphology directly impact DTW matching.
2. **Class Imbalance in Test Recordings:** 6 classes have 10 recorded samples, 1 class has 6 samples, and 18 classes have only 1 sample. Overall accuracy reflects this sample distribution.
3. **Natural-Length DTW Cap:** As observed in repository diagnostics, sequences over 60 frames are resampled. Hand presence flag interpolation on sequences with duration > 60 frames can cause non-binary flags.
4. **Uncalibrated Confidence:** Similarity scores are computed via $s = 1 / (1 + d)$ and are not calibrated Bayesian posterior probabilities.