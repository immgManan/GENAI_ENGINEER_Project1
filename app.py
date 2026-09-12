import streamlit as st
import pandas as pd

from rag_chain import (
    load_vectorstore,
    get_relevant_tickets,
    generate_answer
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Customer Support Copilot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 AI Customer Support Copilot")

st.markdown(
    """
    Use historical customer-support tickets to get AI-powered
    recommendations for customer problems.

    **Technology:** LangChain • RAG • Chroma • LLM • Streamlit
    """
)

st.divider()


# ============================================================
# FILE PATH
# ============================================================

CSV_PATH = "data/customer_support_tickets.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(CSV_PATH)

    # Clean product names
    df["Product Purchased"] = (
        df["Product Purchased"]
        .astype(str)
        .str.strip()
    )

    return df


df = load_data()


# ============================================================
# LOAD CHROMA VECTOR DATABASE
# ============================================================

@st.cache_resource
def load_chroma():

    vectorstore = load_vectorstore()

    return vectorstore


vectorstore = load_chroma()


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    return (
        str(text)
        .lower()
        .strip()
        .replace(" ", " ")
    )


# ============================================================
# GET UNIQUE PRODUCTS
# ============================================================

products = sorted(
    df["Product Purchased"]
    .dropna()
    .unique()
    .tolist()
)


# ============================================================
# PRODUCT SEARCH
# ============================================================

st.subheader("1️⃣ Select Your Product")

st.markdown(
    "Start typing your product name to see available products."
)


# ============================================================
# SEARCH INPUT
# ============================================================

product_search = st.text_input(
    "🔍 Search Product",
    placeholder="Example: Samsung, iPhone, Laptop...",
    key="product_search"
)


# ============================================================
# NORMALIZE SEARCH INPUT
# ============================================================

search_text = normalize_text(product_search)


# ============================================================
# LIVE PRODUCT MATCHING
# ============================================================

if search_text:

    matching_products = [
        product
        for product in products
        if search_text in normalize_text(product)
    ]

    # ========================================================
    # PRODUCTS FOUND
    # ========================================================

    if matching_products:

        st.caption(
            f"🔎 {len(matching_products)} matching product(s) found"
        )

        # ----------------------------------------------------
        # PRODUCT SELECTION
        # ----------------------------------------------------

        selected_product = st.selectbox(
            "Choose your product",
            matching_products,
            key="selected_product"
        )


        # ====================================================
        # PRODUCT INFORMATION
        # ====================================================

        product_data = df[
            df["Product Purchased"] == selected_product
        ]

        ticket_count = len(product_data)


        # ----------------------------------------------------
        # PRODUCT AVAILABILITY CONFIRMATION
        # ----------------------------------------------------

        st.success(
            f"✅ **{selected_product}** is available "
            f"in the support knowledge base."
        )

        st.info(
            f"📚 We have **{ticket_count:,} historical "
            f"support tickets** for this product."
        )


        # ====================================================
        # PRODUCT STATISTICS
        # ====================================================

        with st.expander("📊 View Product Statistics"):

            col1, col2, col3 = st.columns(3)


            # ------------------------------------------------
            # TOTAL TICKETS
            # ------------------------------------------------

            with col1:

                st.metric(
                    "Product Tickets",
                    f"{ticket_count:,}"
                )


            # ------------------------------------------------
            # MOST COMMON ISSUE
            # ------------------------------------------------

            with col2:

                if not product_data.empty:

                    most_common_type = (
                        product_data["Ticket Type"]
                        .value_counts()
                        .index[0]
                    )

                    st.metric(
                        "Most Common Issue",
                        most_common_type
                    )


            # ------------------------------------------------
            # HIGH PRIORITY TICKETS
            # ------------------------------------------------

            with col3:

                high_priority_count = len(
                    product_data[
                        product_data["Ticket Priority"]
                        .astype(str)
                        .str.lower()
                        == "high"
                    ]
                )

                st.metric(
                    "High Priority Tickets",
                    f"{high_priority_count:,}"
                )


        st.divider()


        # ====================================================
        # CUSTOMER PROBLEM
        # ====================================================

        st.subheader("2️⃣ Describe Your Problem")

        question = st.text_area(
            "What problem are you experiencing?",
            placeholder=(
                f"Example: My {selected_product} "
                "is not working properly. "
                "What should I do?"
            ),
            height=150,
            key="customer_question"
        )


        # ====================================================
        # ASK AI BUTTON
        # ====================================================

        ask_button = st.button(
            "🤖 Ask AI Support Assistant",
            type="primary",
            use_container_width=True
        )


        # ====================================================
        # PROCESS QUESTION
        # ====================================================

        if ask_button:

            # ------------------------------------------------
            # VALIDATE QUESTION
            # ------------------------------------------------

            if not question.strip():

                st.warning(
                    "⚠️ Please describe your problem first."
                )

            else:

                # ============================================
                # RETRIEVE RELEVANT TICKETS
                # ============================================

                with st.spinner(
                    "🔎 Searching historical support tickets..."
                ):

                    relevant_docs = get_relevant_tickets(
                        vectorstore=vectorstore,
                        question=question,
                        product=selected_product,
                        k=5
                    )


                # ============================================
                # NO RELEVANT TICKETS
                # ============================================

                if not relevant_docs:

                    st.warning(
                        "⚠️ No relevant historical tickets "
                        "were found for this product."
                    )

                    st.info(
                        """
                        The product exists in the knowledge base,
                        but no sufficiently similar historical
                        support case was found.

                        Try describing the problem using different
                        words or provide more details.
                        """
                    )


                # ============================================
                # RELEVANT TICKETS FOUND
                # ============================================

                else:

                    # ========================================
                    # GENERATE AI ANSWER
                    # ========================================

                    with st.spinner(
                        "🤖 Generating AI recommendation..."
                    ):

                        answer = generate_answer(
                            question=question,
                            product=selected_product,
                            documents=relevant_docs
                        )


                    # ========================================
                    # DISPLAY AI RESPONSE
                    # ========================================

                    st.subheader(
                        "3️⃣ AI Support Recommendation"
                    )

                    st.markdown(answer)

                    st.divider()


                    # ========================================
                    # SOURCE TICKETS
                    # ========================================

                    st.subheader(
                        "📚 Historical Tickets Used"
                    )

                    st.caption(
                        "These support tickets were retrieved "
                        "from Chroma using semantic similarity "
                        "and the selected product."
                    )


                    # ========================================
                    # DISPLAY RETRIEVED DOCUMENTS
                    # ========================================

                    for index, document in enumerate(
                        relevant_docs,
                        start=1
                    ):

                        ticket_id = document.metadata.get(
                            "ticket_id",
                            "Unknown"
                        )

                        priority = document.metadata.get(
                            "priority",
                            "Unknown"
                        )

                        status = document.metadata.get(
                            "status",
                            "Unknown"
                        )


                        with st.expander(
                            f"🎫 Ticket {index} — #{ticket_id}"
                        ):

                            st.caption(
                                f"Priority: {priority} | "
                                f"Status: {status}"
                            )

                            st.write(
                                document.page_content
                            )


    # ========================================================
    # NO PRODUCT FOUND
    # ========================================================

    else:

        st.error(
            f"❌ No product found for **'{product_search}'**."
        )

        st.caption(
            "Try entering another part of the product name."
        )


# ============================================================
# INITIAL STATE
# ============================================================

else:

    st.info(
        "👆 Start typing to search available products."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Customer Support Copilot | "
    "LangChain + RAG + Chroma + LLM + Streamlit"
)