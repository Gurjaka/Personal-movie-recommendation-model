# Movie Recommendation System: Project Report and Overview

??? note "🇬🇪 ქართული ვერსია (Georgian Version)"

    ## ფილმების რეკომენდაციის სისტემა: პროექტის რეპორტი და მიმოხილვა

        ჩვენ შევქმენით ჰიბრიდული ფილმების რეკომენდაციის სისტემა, რომელიც აერთიანებს შინაარსზე დაფუძნებულ ფილტრაციას ფილმის ჟანრების გამოყენებით და კოლაბორაციულ ფილტრაციას მომხმარებლის შეფასებების გამოყენებით პერსონალიზებული ფილმების რეკომენდაციების შესათავაზებლად.

    ### სისტემის კომპონენტები

    #### 1. მონაცემთა დამუშავება

    სისტემა იტვირთავს და წინასწარ ამუშავებს ფილმებისა და რეიტინგის მონაცემებს. იგი იყენებს TF-IDF-ს და კოსინუსურ მსგავსებას შინაარსზე დაფუძნებული რეკომენდაციებისთვის.

    #### 2. მოდელის ტრენინგი

    იქმნება ჰიბრიდული მოდელი, რომელიც აერთიანებს:
    - შინაარსზე დაფუძნებულ ფილტრაციას
    - ჟანრის მსგავსებას
    - კოლაბორაციულ ფილტრაციას
    - FAISS-ს მსგავსებების ეფექტური ძიებისთვის

    სისტემა ინახავს გაწვრთნილ მოდელს ხელახალი გამოყენებისთვის.

    #### 3. ტესტირება და გამართვა

    სისტემა დასტურდება რეკომენდაციებს სატესტო შემთხვევებით და შემდგომ სწორდება მონაცემთა პრობლემები, მაგალითად შეუსაბამო ფილმების სათაურები.

    #### 4. ვიზუალიზაცია

    სისტემა გენერირებს ინფორმაციას, როგორებიცაა:
    - ჟანრების განაწილება
    - რეიტინგის ტენდენციები
    - ყველაზე მაღალი რეიტინგის მქონე ფილმები

    #### 5. მომხმარებლის ინტერფეისი

    Gradio-ს გამოყენებით შეიქმნა მარტივი ურთიერთქმედების ინტერფეისი. მომხმარებელს შეუძლია შეიყვანოს მოწონებული ფილმი და სისტემა რეკომენდაციებს დაუბრუნებს.

    ### ძირითადი მიგნებები

    #### 1. ჰიბრიდული მოდელები აჯობებენ ერთობლივ მიდგომებს

    - **შინაარსზე დაფუძნებული ფილტრაცია** კარგად მუშაობს ცივი დაწყების პრობლემებზე (ახალი მომხმარებლები), მაგრამ აკლია პერსონალიზაცია
    - **კოლაბორაციული ფილტრაცია** აუმჯობესებს პერსონალიზაციას, მაგრამ ვერ მუშაობს, თუ მომხმარებლის მონაცემები მწირია
    - **ორივეს კომბინაცია** (ჰიბრიდული მოდელი) საუკეთესო შედეგს იძლევა

    #### 2. მონაცემთა ხარისხი მნიშვნელოვანია

    - ფილმების სათაურების შესატყვისება რთული იყო (მაგ., "The Dark Knight" vs. "Dark Knight, The")
    - პრობლემა გადაიჭრა `difflib`-ის გამოყენებით უზუსტო შესატყვისებისთვის
    - მოგვიწია რეიტინგების მონაცემების ქვესემპლირება ეფექტურობის გასაუმჯობესებლად

    #### 3. დებაგი აუცილებელია

    დავამატეთ დებაგის ლოგები (`Debug: ...`) რომ ვთვალყუროთ:
    - როგორ იყო მომხმარებლის შეყვანა დამუშავებული
    - რომელი ფილმები იყო შესატყვისი
    - რატომ წარმოიქმნა რეკომენდაციები

    #### 4. FAISS აჩქარებს მსგავსების ძიებას

    შევცვალეთ Scikit-learn-ის `cosine_similarity` FAISS-ით უფრო სწრაფი უახლოესი-მეზობლის ძიებისთვის კოლაბორაციულ ფილტრაციაში.

    #### 5. Gradio აადვილებს განვითარებას

    - ავაშენეთ მარტივი UI <50 ხაზით კოდში
    - მომხმარებლებს შეუძლიათ შეიყვანონ ფილმები და მიიღონ რეკომენდაციები მყისიერად

    ### შეჯამება

    ჰიბრიდული ფილმების რეკომენდაციის სისტემა წარმატებით აერთიანებს შინაარსზე დაფუძნებულ და კოლაბორაციულ ფილტრაციას, რაც უზრუნველყოფს როგორც ახალი მომხმარებლების, ისე გამოცდილი მომხმარებლების მაღალი ხარისხის რეკომენდაციებს. სისტემა ოპტიმიზებულია ეფექტურობისთვის და მარტივია გამოყენებისთვის.

??? note "🇺🇸 English Version"

    ## Movie Recommendation System: Project Report and Overview

    We developed a hybrid movie recommendation system that combines content-based filtering using movie genres and collaborative filtering using user ratings to provide personalized movie recommendations.

    ### System Components
    
    #### 1. Data Processing

    The system loads and preprocesses movie and rating data. It uses TF-IDF and cosine similarity for content-based recommendations.

    #### 2. Model Training

    A hybrid model is created that combines:
    - Content-based filtering
    - Genre similarity
    - Collaborative filtering
    - FAISS for efficient similarity search

    The system saves the trained model for reuse.

    #### 3. Testing and Debugging

    The system validates recommendations with test cases and subsequently fixes data issues, such as inconsistent movie titles.

    #### 4. Visualization

    The system generates information such as:
    - Genre distribution
    - Rating trends
    - Highest-rated movies

    #### 5. User Interface

    A simple interaction interface was created using Gradio. Users can input their favorite movies and the system returns recommendations.

    ### Key Findings
    
    #### 1. Hybrid Models Outperform Individual Approaches

    - **Content-based filtering** works well for cold start problems (new users) but lacks personalization
    - **Collaborative filtering** improves personalization but fails when user data is sparse
    - **Combining both** (hybrid model) provides the best results

    #### 2. Data Quality is Critical

    - Movie title matching was challenging (e.g., "The Dark Knight" vs. "Dark Knight, The")
    - The problem was solved using `difflib` for fuzzy matching
    - We had to subsample rating data to improve efficiency

    #### 3. Debugging is Essential

    We added debug logs (`Debug: ...`) to monitor:
    - How user input was processed
    - Which movies were matched
    - Why recommendations were generated

    #### 4. FAISS Accelerates Similarity Search

    We replaced Scikit-learn's `cosine_similarity` with FAISS for faster nearest-neighbor search in collaborative filtering.

    #### 5. Gradio Simplifies Development

    - Built a simple UI in <50 lines of code
    - Users can input movies and receive recommendations instantly

    ### Conclusion

    The hybrid movie recommendation system successfully combines content-based and collaborative filtering, ensuring high-quality recommendations for both new and experienced users. The system is optimized for efficiency and easy to use.
