from streamlit_option_menu import option_menu
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

st.set_page_config(
    page_title="Klasifikasi Penyakit Daun Tomat",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASS_NAMES = [
    "Tomato Healthy",
    "Tomato Late Blight",
    "Tomato Mosaic Virus",
    "Tomato Septoria Leaf Spot",
    "Tomato Yellow Leaf Curl Virus",
]

IMG_SIZE = (224, 224)
MODEL_PATH = r"klasifikasitomat.h5"

INFO_PENYAKIT = {
    "Tomato Healthy": {
        "deskripsi": "Daun tomat dalam kondisi sehat.",
        "karakteristik": "Daun berwarna hijau tua atau muda cerah, permukaan daun halus, daunnya kuat.",
        "REKOMENDASI": [
            "Siram tanaman tomat secara rutin dan teratur.",
            "Memangkas daun yang sudah tua secara berkala.",
            "Tidak menyentuh tanaman sehat setelah menyentuh tanaman terinfeksi.",
        ],
    },
    "Tomato Late Blight": {
        "deskripsi": (
            "Penyakit Late Blight disebabkan oleh cendawan Phytophthora infestans dan berkembang pada kondisi yang basah serta dingin dan umumnya terjadi di daerah dataran tinggi (Setiawati et al., 2001). "
            "Selain itu, spora Phytophthora infestans berkembang cepat pada suhu 10-25° C dengan kelembapan lebih dari 75% selama 2 hari atau lebih, terutama jika daun basah dan berembun. "
            "Sporangia yang terinfeksi dapat menyebar melalui angin maupun percikan air (Alviani Munif, 2017)."
        ),
        "karakteristik": (
            "Bercak cokelat kehitaman tidak beraturan dengan kondisi daun tampak basah, lunak, dan berwarna kehitam-hitaman. "
            "Dalam kondisi udara lembab, bercak-bercak akan bertambah besar dan bagian yang terserang akan membusuk serta mengeluarkan bau yang tidak sedap. "
            "Tumbuh bulu atau spora putih di bagian bawah daun. "
            "Dalam kondisi udara basah penyakit busuk daun dapat menyebar sangat cepat ke seluruh dedaunan yang membuat daun layu serta mati dengan cepat."
        ),
        "penanganan": [
            "Menggunakan pelindung atau naungan hujan untuk mengurangi serangan penyakit (Alviani Munif, 2017).",
            "Mengaplikasikan fungisida yang efektif baik bekerja dari luar (kontak) seperti Acylalamine, Oxadityl, atau Propamocarb serta fungisida yang bekerja dari dalam (sistemik) tanaman seperti Clorotaloni secara bergiliran (Balai Pengkajian Teknologi Pertanian Kalimantan Selatan, 2010).",
            "Segera cabut daun yang terinfeksi atau musnahkan pohon tomat apabila serangan sudah lebih dari 60% (Nurhanipah & Supriatna, 2025).",
            "Menggunakan jamur endofitik Trichoderma spp. pada media tanam untuk menghambat pertumbuhan Phytophthora infestans dan mengurangi intensitas penyakit (Wattimury et al., 2021).",
        ],

        "REKOMENDASI": [
            "Tanam varietas tanaman yang tahan terhadap penyakit busuk daun.",
            "Membersihkan tanaman liar atau sisa-sisa tanaman sebelumnya.",
            "Kurangi kelembaban kebun dengan meningkatkan jarak tanam (Gevens & Wilbur, 2015).",
            "Gunakan benih berkualitas.",
            "Hindari penanaman saat musim hujan, lakukan saat musim kemarau.",
            "Lakukan rotasi tanaman dan menjaga kebersihan peralatan berkebun.",
        ],
    },
    "Tomato Mosaic Virus": {
        "deskripsi": (
            "Tomato Mosaic Virus disebabkan oleh Tomato Mocaic Virus (ToMV) dari genus tobamovirus dan famili Virgaviridae yang penyebarannya melalui kontak dengan tanaman yang terinfeksi baik pada tangan, pakaian, maupun alat pertanian (Wati et al., 2021). "
            "ToMV dapat bertahan dalam benih yang terinfeksi hingga 10 tahun serta tetap bertahan pada sisa tanaman maupun permukaan benda dalam waktu yang lama sehingga berpotensi menjadi sumber infeksi baru (Panno et al., 2021)."
        ),
        "karakteristik": (
            "Perubahan warna pada permukaan daun berupa kombinasi hijau terang dan gelap yang tidak merata seperti pola mozaik "
            "terkadang disertai klorosis berwarna kuning (Majid & Ariatmanto, 2025)."
        ),
        "penanganan": [
            "Mencabut dan memusnahkan tanaman yang terinfeksi (Wati et al., 2021).",
            "Musnahkan dan tidak mengompos sisa tanaman yang terinfeksi.",
            "Sterilisasi alat pertanian dan mencuci tangan setelah menangani tanaman yang terinfeksi (Schuh et al., 2021)."

        ],
        "REKOMENDASI": [
            "Menggunakan varietas tomat yang tahan terhadap Tomato Mosaic Virus (Dombrovsky & Smith, 2017).",
            "Menggunakan bibit yang bersertifikat bebas patogen.",
            "Sterilisasi alat pertanian sebelum dan sesudah digunakan.",
            "Merotasi tanaman dan membersihkan sisa-sisa tanaman maupun gulma yang dapat menjadi inang virus secara rutin (Panno et al., 2021)."
        ],
    },
    "Tomato Septoria Leaf Spot": {
        "deskripsi": (
            "Septoria Leaf Spot adalah penyakit yang disebabkan oleh jamur Septoria Lycopersici Speg yang bertahan pada sisa tanaman dan tanah."
            "Sporanya menyebar melalui percikan air hujan/irigasi dan dapat berkembang cepat pada kelembaban tinggi dengan temperatur +- 20°C dalam waktu yang lama (Agus, 2021)."
        ),
        "karakteristik": (
            "Bintik-bintik gelap pada daun, tepi bintik berwarna cokelat atau abu-abu dengan tepi yang lebih gelap. "
            "Jamur ini menyerang bagian bawah daun terlebih dahulu kemudian perlahan menyebar ke daun bagian atas, lalu daun menguning dan rontok."
        ),
        "penanganan": [
            "Menyemprotkan fungisida seperti Zineb dan Maneb.",
            "Menyemprotkan tembaga oxychlorida sebanyak 500 -750 gram ke dalam 100 liter air. Agar lebih efektif dapat ditambahkan Thiophanatemethyl (70% WP).",
            "Buang dan musnahkan daun yang terinfeksi tidak dikubur dalam tanah (Agus, 2021)",
            "Gunakan mulsa untuk mencegah spora bercak daun septoria di tanah mencapai daun.",
        ],
        "REKOMENDASI": [
            "Rotasi tanaman dengan menanam tanaman lain yang berbeda family (bukan tanaman Solanaceae).",
            "Menanam varietas tomat yang resisten terhadap penyakit.",
            "Menjaga sanitasi kebun dan bersihkan sisa tanaman setelah panen.",
            "Meningkatkan sirkulasi udara dengan mengatur jarak antar tanaman.",
            "Gunakan bedengan yang lebih tinggi dan lakukan rotasi bedengan setiap tahun agar tanah tidak terus terpapar penyakit.",
        ],
    },
    "Tomato Yellow Leaf Curl Virus": {
        "deskripsi": (
            "Tomato Yellow Leaf Curl Virus (TYLCV) adalah penyakit yang disebakan oleh virus gemini dari genus Begomovirus yang ditularkan oleh kutu kebul. "
            "Jika satu tanaman terinfeksi virus kuning, kemungkinan besar dapat "
            "menular ke sekitar pohon tomat lainnya. Hal ini menyebabkan penyebaran virus kuning "
            "dianggap sebagai potensi ancaman yang cukup serius terhadap budidaya tomat (Wati et al., 2021)."
        ),
        "karakteristik": (
            "Daun muda menggulung ke atas dan menguning, "
            "bentuk daun mengkerut atau keriting, "
            "ukuran daun mengecil, "
            "tanaman kerdil, bunga rontok sebelum berkembang."
        ),
        "penanganan": [
            "Memusnahkan tanaman yang terinfeksi (Wati et al., 2021).",
            "Kendalikan kutu kebul dengan insektisida Imidakloprid dan nimba yang disemprotkan ke tanah.",
            "Gunakan plastik yang memantulkan cahaya / mulsa jerami untuk melindungi tanaman tomat dari kutu kebul.",
            "Semprotkan pestisida sistemik pada tanaman tomat sehat dan sakit untuk mengendalikan penyebaran kutu kebul (Alviani Munif, 2017)."
        ],
        "REKOMENDASI": [
            "Menggunakan varietas tanaman tomat yang tahan terhadap serangga kutu kebul dan virus gemini.",
            "Menjaga kondisi lingkungan tanaman tetap bersih dan jauh dari tanaman inang atau gulma agar terhindar dari hama (Wati et al., 2021).",
            "Membuang sisa tanaman tomat sehat dan sakit setelah panen berakhir dan membakar sisa tanaman (Alviani Munif, 2017).",
            "Menjaga kebersihan tangan serta alat berkebun.",
            "Menyiram tanaman secara rutin dan cukup.",
        ],
    },
}

# ── Model ──────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    return None


def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def predict(model, img_array):
    preds = model.predict(img_array, verbose=0)
    idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][idx]) * 100
    return CLASS_NAMES[idx], confidence, preds[0]

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>Navigasi</h1>", unsafe_allow_html=True)
    menu = option_menu(
        menu_title=None,
        options=["Beranda", "Klasifikasi", "Tentang"],
        icons=["house", "search", "info-circle"],
        default_index=0,
    )

# ── Halaman Beranda ────────────────────────────────────────────────────────────

if menu == "Beranda":
    st.markdown(
        "<h1 style='text-align: center;'>🍅 Website Klasifikasi Penyakit Daun Tomat 🍅</h1>",
        unsafe_allow_html=True,
    )
    st.header("Halo!")
    st.markdown("Tomat adalah salah satu tanaman yang paling banyak dikonsumsi dan dibudidayakan di Indonesia. Penyakit pada daun tomat sering kali tidak disadari sejak awal dan dapat menyebar dengan cepat hingga merusak tanaman. Website ini dapat digunakan untuk membantu mengenali 5 kondisi daun tomat beserta informasi karakteristik, penanganan, dan saran perawatannya.")

    img = Image.open("Tomat Header.jpg")
    st.image(img, use_container_width=True)
    st.divider()

    st.subheader("Informasi Penyakit Daun Tomat")

    penyakit_list = [
    {
        "nama": "Daun Sehat",
        "gambar": "gambar/Sehat.jpg",
        "deskripsi": "Daun tomat berada dalam kondisi sehat tanpa karakteristik penyakit sehingga pertumbuhan tanaman dapat berlangsung secara normal."
    },
    {
        "nama": "Busuk Daun",
        "gambar": "gambar/Late Blight.jpg",
        "deskripsi": "Penyakit yang disebabkan oleh Phytophthora infestans dan dapat berkembang dengan cepat pada kondisi lembap."
    },
    {
        "nama": "Virus Mosaik",
        "gambar": "gambar/Mosaic.jpg",
        "deskripsi": "Penyakit akibat infeksi Tomato Mosaic Virus (ToMV) yang menyebabkan perubahan warna daun menjadi belang seperti pola mosaik."
    },
    {
        "nama": "Bercak Daun Septoria",
        "gambar": "gambar/Septoria.jpg",
        "deskripsi": "Penyakit yang disebabkan oleh jamur *Septoria lycopersici* dan ditandai dengan munculnya bercak kecil pada daun."
    },
    {
        "nama": "Virus Kuning Keriting",
        "gambar": "gambar/YLCV.jpg",
        "deskripsi": "Penyakit akibat infeksi Tomato Yellow Leaf Curl Virus (TYLCV) yang menyebabkan daun menguning, menggulung, dan pertumbuhan tanaman terhambat."
    },
]

    cols = st.columns(3)

    for i, penyakit in enumerate(penyakit_list):
        with cols[i % 3]:
            img = Image.open(penyakit["gambar"]).resize((400, 220))
            st.image(img, use_container_width=True)

            st.markdown(
                f"<p style='text-align:center; font-size:16px; font-weight:bold;'>"
                f"{penyakit['nama']}</p>",
                unsafe_allow_html=True,
            )

            with st.expander("Deskripsi Singkat"):
                st.write(penyakit["deskripsi"])

    st.divider()
    st.subheader("Cara Menggunakan Website")
    st.markdown("1. Klik halaman klasifikasi pada sidebar.")
    st.markdown("2. Unggah gambar daun yang ingin diklasifikasikan secara jelas.")
    st.markdown("3. Klik tombol klasifikasi untuk melihat hasil klasifikasi model beserta cara penanganan dan rekomendasi.")

# ── Halaman Klasifikasi ────────────────────────────────────────────────────────

elif menu == "Klasifikasi":
    st.markdown(
        "<h1 style='text-align: center;'>Klasifikasi Penyakit Daun Tomat 🔎</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "Ayo, upload foto daun tomat untuk mengetahui jenis penyakit dan penjelasannya!"
    )
    st.divider()

    model = load_model()

    if "show_result" not in st.session_state:
        st.session_state.show_result = False

    if "last_file" not in st.session_state:
        st.session_state.last_file = None

    uploaded_file = st.file_uploader(
        "Unggah gambar daun tomat yang ingin diklasifikasi",
        type=["jpg", "jpeg", "png"],
        help="Format yang didukung adalah JPG, JPEG, dan PNG.",
    )

    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.last_file:
            st.session_state.last_file = uploaded_file.name
            st.session_state.show_result = False

        img = Image.open(uploaded_file)

        col_prev, _ = st.columns([1, 1])
        with col_prev:
            st.image(
                img,
                caption=uploaded_file.name,
                use_container_width=True,
            )

            if not st.session_state.show_result:
                if st.button(
                    "Klasifikasi",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.show_result = True
                    st.rerun()

    else:
        st.info(
            "Belum ada gambar yang diunggah. Silakan unggah gambar terlebih dahulu!"
        )

    # Menampilkan hasil
    if uploaded_file and st.session_state.show_result:
        st.divider()
        st.markdown("### Hasil Klasifikasi")

        if model is None:
            st.error("Model tidak ditemukan! Periksa kembali path model.")
        else:
            with st.spinner("Model sedang menganalisis gambar..."):
                img_array = preprocess_image(img)
                label, confidence, all_probs = predict(model, img_array)

            info = INFO_PENYAKIT[label]

            if label == "Tomato Healthy":
                st.success(f"### {label}")
            else:
                st.error(f"### {label}")

            st.info(f"Tingkat Keyakinan Model: {confidence:.2f}%")

            if label != "Tomato Healthy":
                st.write(info["deskripsi"])
            st.divider()

            st.markdown("**KARAKTERISTIK**")
            st.info(info["karakteristik"])
            st.divider()

            if label == "Tomato Healthy":
                st.markdown("**REKOMENDASI**")
                with st.container(border=True):
                    for item in info["REKOMENDASI"]:
                        st.markdown(f"- {item}")
            else:
                col_h, col_r = st.columns(2)

                with col_h:
                    st.markdown("**PENANGANAN**")
                    with st.container(border=True):
                        for i, step in enumerate(info["penanganan"], 1):
                            st.markdown(f"**{i}.** {step}")

                with col_r:
                    st.markdown("**REKOMENDASI**")
                    with st.container(border=True):
                        for item in info["REKOMENDASI"]:
                            st.markdown(f"- {item}")

# ── Halaman Tentang ────────────────────────────────────────────────────────────

elif menu == "Tentang":
    st.markdown(
        "<h1 style='text-align: center;'>Tentang Website</h1>",
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown(
        "Website ini mennggunakan model berbasis Convolutional Neural Network (CNN) "
        "dengan arsitektur MobileNetV2 yang telah dilatih menggunakan dataset daun tomat untuk mengklasifikan kondisi daun tomat berdasarkan citra."
    )

    st.markdown("**Website ini bertujuan untuk:**")
    st.markdown(
        """
        - Membantu petani maupun masyarakat dalam mengenali kemungkinan penyakit pada daun tomat.
        - Menyajikan hasil klasifikasi sebagai referensi awal mengenai kondisi daun tomat dan bukan sebagai pengganti diagnosa ahli.
        - Menjadi media edukasi mengenai penyakit yang dapat menyerang daun tanaman tomat.
        """
    )

    st.divider()
    st.write("Shalwa Rahgiant Permata Putri - 2026")