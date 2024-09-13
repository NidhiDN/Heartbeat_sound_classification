import streamlit as st
import argparse
from fpdf import FPDF
from io import BytesIO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicted_category", type=str)
    args = parser.parse_args()

    class_name = args.predicted_category
    label_info = {
        "normal": {
            "name": "Normal",
            "description": "A normal heartbeat, known as normal sinus rhythm (NSR), is characterized by the synchronized electrical activity of the heart, originating from the sinoatrial (SA) node. This electrical impulse triggers rhythmic contractions of the heart chambers, ensuring efficient blood circulation throughout the body. In NSR, the heart rate typically ranges between 60 to 100 beats per minute in adults at rest, with a steady rhythm and consistent intervals between each heartbeat. During auscultation, normal heart sounds comprise the distinct lub-dub pattern, representing the closure of heart valves during systole and diastole. Variations in heart rate are influenced by factors like age, physical activity, and stress levels, with the body's autonomic nervous system regulating heart rate accordingly. Monitoring heart rate and rhythm is crucial for assessing cardiovascular health, as deviations from normal sinus rhythm may indicate underlying cardiac conditions and warrant further evaluation by healthcare professionals.",
            "diagnosis": "A normal heart sound indicates a healthy heart with no abnormalities detected.",
            "treatments": "No specific treatment is needed for a normal heart sound. However, maintaining a healthy lifestyle with regular exercise and balanced diet is recommended."
        },
        "artifact": {
            "name": "Artifact",
            "description": "Artifacts in ECG recordings can result from a variety of factors, including poor electrode contact, patient movement, electrical interference, or technical errors during recording or processing. They can distort the appearance of the ECG waveform, making interpretation challenging.",
            "diagnosis": "Identifying artifacts involves careful examination of the ECG tracing, considering technical factors during recording, and comparing the findings with the patient's clinical status. Repeating the ECG under controlled conditions may help differentiate true cardiac abnormalities from artifacts.",
            "treatments": "Treatment involves optimizing recording conditions to minimize artifacts. This may include ensuring proper electrode placement, reducing patient movement during recording, using high-quality recording equipment, and troubleshooting technical issues. Addressing underlying medical conditions that may contribute to artifact generation is also important."
        },
        "murmur": {
            "name": "Murmur",
            "description": "murmurs are common in healthy individuals, particularly children, and often do not signify any underlying heart disease. They may be heard during periods of rapid growth or fever and tend to disappear with age. Pathological murmurs, on the other hand, can indicate various heart conditions such as valve disorders (e.g., mitral valve prolapse, aortic stenosis), congenital heart defects, or problems with the heart muscle (cardiomyopathy).",
            "diagnosis": "During a physical examination, the healthcare provider listens to the heart using a stethoscope placed on different areas of the chest. Murmurs are graded based on their intensity, timing, and location. Additional tests like echocardiography, ECG, or chest X-ray help determine the cause and severity of the murmur.",
            "treatments": "Treatment depends on the underlying cause. Innocent murmurs may not require any treatment beyond regular monitoring. Pathological murmurs may necessitate medication to manage symptoms or underlying conditions, surgical repair or replacement of damaged heart valves, or other interventions as indicated."
        },
        "extrasystole": {
            "name": "Extrasystole",
            "description": "Extrasystoles are premature heartbeats that occur outside the normal rhythm. They can originate from various locations in the heart's conduction system and may be benign or symptomatic, depending on frequency and associated symptoms.",
            "diagnosis": "Diagnosis involves recording the heart's electrical activity using an ECG. Extrasystoles appear as premature or abnormal beats on the ECG tracing, occurring before the expected next normal heartbeat.",
            "treatments": "Treatment may not be necessary for occasional or isolated extrasystoles without underlying heart disease. However, if extrasystoles are frequent, symptomatic, or associated with underlying heart conditions, treatment may include medications (such as beta-blockers or antiarrhythmic drugs) to regulate heart rhythm or address underlying heart conditions. Lifestyle modifications such as reducing stress, caffeine, and alcohol intake may also be recommended."
        },
        "extrahls": {
            "name": "Extrahls",
            "description": " Extrahls could potentially refer to additional heart sounds beyond the normal S1 and S2 sounds. These additional sounds, such as a third heart sound (S3) or a fourth heart sound (S4), can indicate abnormal cardiac function and may be associated with conditions like heart failure, myocardial infarction, or valvular heart disease.",
            "diagnosis": " Diagnosis involves auscultation of the heart sounds using a stethoscope during a physical examination. Additional diagnostic tests such as echocardiography or cardiac imaging may be performed to evaluate the underlying cause of the extra heart sound.",
            "treatments": "Treatment depends on the underlying cause of the extra heart sound. For example, if it's due to heart failure, treatment may involve medications to improve heart function (such as ACE inhibitors, beta-blockers, or diuretics), lifestyle modifications (such as salt restriction and regular exercise), and monitoring of fluid intake and weight. In cases of valvular heart disease, treatment may involve medications, repair, or replacement of the affected valve, or other interventions as deemed appropriate by a cardiologist."
        },
    }

    details = label_info.get(class_name)
    if details:
        st.title(f"{details['name']} Information")
        st.write(f"Description: {details['description']}")
        st.write(f"Diagnosis: {details['diagnosis']}")
        st.write(f"Treatments: {details['treatments']}")

        if st.button("Download PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            # Add title with styling
            pdf.set_font("Arial", style="B", size=16)
            pdf.cell(200, 10, txt=f"{details['name']} Information", ln=True, align="C")
            pdf.set_font("Arial", size=12)  # Reset font
            pdf.ln(14)

            # Add description with styling
            pdf.set_font("Arial", style="I", size=12)
            pdf.multi_cell(0, 10, txt=f"<i>Description:</i> {details['description']}", align="L")
            pdf.ln(10)

            # Add diagnosis with styling
            pdf.multi_cell(0, 10, txt=f"<i>Diagnosis:</i> {details['diagnosis']}", align="L")
            pdf.ln(10)
            # Add treatments with styling
            pdf.multi_cell(0, 10, txt=f"<i>Treatments:</i> {details['treatments']}", align="L")
            pdf
            pdf_bytes = pdf.output(dest="S").encode("latin1")

            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"{details['name']}_info.pdf",
                mime="application/pdf"
            )

    else:
        st.write("Details not available.")

if __name__ == "__main__":
    main()
