class Constants:
    @staticmethod
    def should_build_kb()-> bool:
        return False

    @staticmethod
    def cleaned_csv_path() ->str:
       return 'data/cleaned_faqs.csv'

    @staticmethod
    def default_kb_count() -> int:
        return (5)

    @staticmethod
    def confidence_for_not_related() -> float:
        return 1.2
