import hashlib
import json
import logging
import os
import re
from datetime import datetime
from typing import TypeVar

import ask_sdk_core.utils as ask_utils
import boto3
import sentry_sdk
from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_core.skill_builder import CustomSkillBuilder
from ask_sdk_dynamodb.adapter import DynamoDbAdapter
from ask_sdk_model import Response
from ask_sdk_model.interfaces.connections import SendRequestDirective
from ask_sdk_model.interfaces.monetization.v1 import PurchaseResult
from ask_sdk_model.services.monetization import EntitledState
from boto3.dynamodb.conditions import Key, Attr
from dotenv import load_dotenv
from openai import OpenAI
from sentry_sdk import capture_exception, Hub, flush
from sentry_sdk.integrations.aws_lambda import AwsLambdaIntegration

LARGE_PACK_QUERIES = 500

MEDIUM_PACK_QUERIES = 150

SMALL_PACK_QUERIES = 50

DEFAULT_QUERIES = 5

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

load_dotenv()  # take environment variables from .env.

sentry_sdk.init(
    dsn="https://73308afe464261cf3ad9756bfdae5fff@o4506961510596608.ingest.us.sentry.io/4507806986862592",
    integrations=[AwsLambdaIntegration()],
    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for tracing.
    traces_sample_rate=1.0,
    # Set profiles_sample_rate to 1.0 to profile 100%
    # of sampled transactions.
    # We recommend adjusting this value in production.
    profiles_sample_rate=1.0,
    shutdown_timeout=0.5  # the timeout is TIGHT, only 8 seconds
)

# DynamoDB setup
ddb_region = os.environ.get('DYNAMODB_PERSISTENCE_REGION')
ddb_table_name = os.environ.get('DYNAMODB_PERSISTENCE_TABLE_NAME')
ddb_resource = boto3.resource('dynamodb', region_name=ddb_region)
dynamodb_adapter = DynamoDbAdapter(table_name=ddb_table_name, create_table=False, dynamodb_resource=ddb_resource)

# Perplexity API setup
PERPLEXITY_API_KEY = os.environ.get('PERPLEXITY_API_KEY')
client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")
MODEL_USED = "llama-3.1-sonar-small-128k-online"

# In-skill product IDs
SMALL_PACK_ID = "amzn1.adg.product.9f423cf3-ce4b-4fa2-b8ba-29ac45eddb36"
MEDIUM_PACK_ID = "amzn1.adg.product.8e41854a-b1a2-4b94-9bfb-e34f5ca7ccf6"
LARGE_PACK_ID = "amzn1.adg.product.cef4094c-3678-4061-8734-b5518637d5ff"
SUBSCRIPTION_ID = "amzn1.adg.product.2f74f637-5cd2-43c3-bbca-0df0f0272b26"

# ====== CONTENT GUIDELINES ======
# Do not include sexually explicit content or pornography
# Do not depict graphic violence, gore, or extreme content
# Do not promote illegal activities, terrorism, or organized crime
# Do not include hate speech or discriminatory content targeting any group
# Do not provide information on forced marriages or human trafficking
# Do not use symbols associated with hate groups
# Do not promote illegal downloading or pirated software
# Do not give advice on joining illegal organizations
# Do not provide guidance on illegal lifestyles
# Do not give instructions for creating dangerous materials (e.g., bombs, drugs)
# Do not promote or praise terrorism
# Do not promote the use, sale, or distribution of illegal drugs
# Do not enable or encourage illegal gambling
# Do not promote tobacco products or underage use of such products
# Do not promote excessive or underage alcohol use
# Do not use excessive profanity
# Do not include false or misleading content
# Do not show insensitivity towards disasters or conflicts
# Do not aim profanity at children
# Do not include malicious content or undisclosed pranks
# Do not produce unmoderated user-generated content
# Do not perform searches that violate privacy or copyright laws
# Do not provide personal information without consent
# Do not engage in or promote cyberbullying or online harassment
# Do not assist in academic dishonesty or plagiarism
# Do not ignore content warnings or age restrictions when applicable

SYSTEM_PROMPT = """
You are Lyra AI, an advanced AI assistant designed to provide concise, accurate answers on a wide range of topics. 
Your responses should be brief, informative, and directly address the user's query without unnecessary filler. 
Maintain a friendly tone while focusing on delivering valuable information in plain text, without technical jargon. 
Draw upon your extensive knowledge to assist with any question, while adhering to ethical guidelines. 
Don't include explicit sexual content, graphic violence, hate speech, or promotion of illegal activities. 
Don't provide instructions for creating weapons, drugs, or other dangerous materials. 
Don't encourage substance abuse, underage drinking, or tobacco use. 
Don't give false or misleading information, and remain sensitive when discussing disasters, conflicts, or tragedies. 
Don't assist with illegal downloads, hacking, or any criminal activities. 
Don't provide personal information of any individual or business, including addreses, phone numbers, zip codes, and financial information including social security numbers and credit card numbers.
Don't share personal information or engage in cyberbullying. 
Don't provide medical diagnoses or legal advice. 
Don't discuss things that are illegal in some parts of the world, like jailbreaking a phone.
Your goal is to be helpful and informative in every interaction, providing relevant answers without citing sources or using markdown like bold markers and headings. 
Keep responses compact, absolutely no more than 5 sentences, and without lists and focusing on complete thoughts. 
Don't refer users to external services; simply answer questions to the best of your ability within these guidelines.
""".replace("\n", " ")

# i18n
from ask_sdk_core.utils import get_locale

LOCALE_TEXTS = {
    "en": {
        "WELCOME": "Hi, I'm Lyra, an advanced AI assistant. I can answer a wide range of questions using advanced AI. Just ask anything you'd like to know.",
        "HELP": "Lyra can answer a wide range of questions using advanced AI. Just ask anything you'd like to know. You have {remaining_queries} queries left. To buy more, say 'buy queries'.",
        "FALLBACK": "I didn't understand that. You can ask me a question or say 'help' for more information.",
        "GOODBYE": "Thank you for using Lyra AI. Goodbye!",
        "ERROR": "I'm having trouble processing your request. Please try again.",
        "QUERY_LIMIT_REACHED": "You've used all your queries. To get more, say 'buy queries'.",
        "REMAINING_QUERIES": "You have {remaining_queries} queries left. What would you like to ask?",
        "BUY_QUERIES_PROMPT": "Lyra AI offers several query packs: a small pack with 50 queries, a medium pack with 150 queries, and a large pack with 500 queries. There's also a monthly unlimited subscription. To purchase one, say 'buy small pack', 'buy medium pack', 'buy large pack', or 'buy subscription'.",
        "PURCHASE_SUCCESSFUL": "Purchase complete! Your queries have been added. What would you like to know?",
        "PURCHASE_FAILED": "The purchase wasn't successful. You can try again or ask a different question.",
        "PREMIUM_OWNED": "You have a subscription to Lyra AI. You have unlimited queries. What would you like to know?",
        "PREMIUM_CANCELLED": "Your subscription has been cancelled. You now have a limited number of queries. What would you like to know?",
        "PREMIUM_CANCELLATION_FAILED": "There was an error cancelling your subscription. Please try again later.",
        "QUERY_PACK_UNAVAILABLE": "In-skill purchasing is not supported in this region.",
        "SMALL_QUERY_PACK_FEEDBACK": "You've purchased a small query pack, which added 50 queries. You have {remaining_queries} left. What would you like to know?",
        "MEDIUM_QUERY_PACK_FEEDBACK": "You've purchased a medium query pack, which added 150 queries. You have {remaining_queries} left. What would you like to know?",
        "LARGE_QUERY_PACK_FEEDBACK": "You've purchased a large query pack, which added 500 queries. You have {remaining_queries} left. What would you like to know?",
        "UNLIMITED_QUERY_PACK_FEEDBACK": "You've purchased a monthly subscription. You now have unlimited queries. What would you like to know?",
        "NO_PURCHASED_ITEM": "You haven't made a purchase yet. To buy queries, say 'buy queries'. You have {remaining_queries} left. What would you like to know?",
        "REFER_CANCELLATION": "To cancel your subscription, please visit the Alexa app. What would you like to know?",
        "EXPLICIT_CONTENT": "I'm sorry, I can't provide that information as it contains explicit content. Would you like to ask something else?",
        "PERSONAL_CONTENT": "I'm sorry, I can't provide that information as it contains personal information. Would you like to ask something else?"
    },
}


def contains_mature_content(text):
    # type: (str) -> bool

    text = text.lower()

    # Profanity words
    profanity = [
        "fuck", "shit", "ass", "bitch", "damn", "cunt", "whore", "bastard", "prick",
        "asshole", "dickhead", "motherfucker", "twat", "bollocks", "wanker", "douchebag"
    ]

    # Sexual content words
    sexual = [
        "sex", "nude", "porn", "xxx", "cock", "pussy", "dick", "boob", "tit",
        "masturbation", "orgasm", "dildo", "condom", "penis", "vagina", "blowjob",
        "anal", "handjob", "bdsm", "fetish", "kink", "lingerie", "semen", "ejaculation"
    ]

    # Hate speech words
    hate_speech = [
        "nigger", "faggot", "retard", "kike", "spic", "chink", "wetback", "raghead", "nazi",
        "homo", "dyke", "tranny", "gook", "paki", "jew", "gypsy", "cripple", "negro"
    ]

    # Violence-related words
    violence = [
        "kill", "murder", "rape", "assault", "terrorist", "gore", "decapitation", "bomb", "weapon",
        "stab", "shoot", "strangle", "torture", "massacre", "slaughter", "execution", "brutality"
    ]

    # Drug-related words
    drugs = [
        "cocaine", "heroin", "meth", "crack", "weed", "drug", "lsd", "marijuana", "ecstasy", "mdma",
        "amphetamine", "opium", "ketamine", "mushroom", "peyote", "steroid", "overdose", "needle"
    ]

    # Gambling-related words
    gambling = [
        "gambling", "bet", "casino", "poker", "roulette", "blackjack", "slot", "lottery",
        "wager", "bookie", "odds", "jackpot", "dice", "baccarat", "keno"
    ]

    # Illegal activity words
    illegal = [
        "pirate", "hack", "theft", "steal", "fraud", "counterfeit", "smuggle", "launder",
        "bribe", "extort", "embezzle", "forgery", "scam", "blackmail", "heist", "jailbreak"
    ]

    # Combine all categories into a single list
    all_words = (
        profanity +
        sexual +
        hate_speech +
        violence +
        drugs +
        gambling +
        illegal
    )

    # Create the pattern using word boundaries
    pattern = r'\b(' + '|'.join(map(re.escape, all_words)) + r')\b'

    if re.search(pattern, text):
        return True
    else:
        return False


def contains_personal_info(text):
    # type: (str) -> bool

    # Convert text to lowercase for case-insensitive matching
    text = text.lower()

    # Patterns for different types of personal information
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'ssn': r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
        'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'address': r'\b\d+\s+(?:[A-Za-z]+\s*)+(?:street|st|avenue|ave|road|rd|highway|hwy|square|sq|trail|trl|drive|dr|court|ct|park|parkway|pkwy|circle|cir|boulevard|blvd)\b',
        'zip_code': r'\b\d{5}(?:-\d{4})?\b',
        # UK format (like 0161-868-8000)
        'uk_pattern': r'^\s*0\d{3}[-\s]?\d{3}[-\s]?\d{4}\s*$',
        # International format with country code (like +34 913 440 557)
        'international_pattern': r'^\s*\+\d{1,3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{3}\s*$',
        # US/Canada format (like (123) 456-7890)
        'us_pattern': r'^\s*(?:\(\d{3}\)|\d{3})[-\s]?\d{3}[-\s]?\d{4}\s*$',
        # General international format (flexible)
        'flexible_international': r'^\s*(?:\+?\d{1,3}[-\s]?)?\d{2,4}[-\s]?\d{3,4}[-\s]?\d{3,4}\s*$',
    }

    # Check for each pattern in the text
    for pattern_name, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


def strip_markdown_and_newlines(text):
    # type: (str) -> str

    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)

    # Remove inline code
    text = re.sub(r'`[^`\n]+`', '', text)

    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)

    # Remove bold and italic
    text = re.sub(r'\*{1,2}([^\*\n]+)\*{1,2}', r'\1', text)

    # Remove links
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove bullet points and numbered lists
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Remove horizontal rules
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)

    # Remove newlines
    text = re.sub(r'\n+', ' ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def get_text(handler_input, key):
    # type: (HandlerInput, str) -> str

    locale = get_locale(handler_input)
    if key in LOCALE_TEXTS.keys():
        return LOCALE_TEXTS[locale][key]
    language = locale.split("-")[0]
    return LOCALE_TEXTS.get(language, LOCALE_TEXTS["en"]).get(key, key)  # for debugging


class LaunchRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        speak_output = get_text(handler_input, "WELCOME")
        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )


def get_hash_of_query(query):
    salted_query = f"{SYSTEM_PROMPT}+{query}"
    query_hash = hashlib.sha256(salted_query.encode()).hexdigest()
    return query_hash


def clean_ddb_table():
    # type: () -> None
    """
    purges the ddb table AJAX style
    """
    # NOTE: doesn't work on alexa-hosted because it lacks the Scan and BatchWrite permissions

    # Get current timestamp
    ttl_expired = int(datetime.now().timestamp())

    # Scan the table for items where the id starts with "query-", attributes is a string, or ttl is expired
    response = ddb_resource.Table(ddb_table_name).scan(
        FilterExpression=(
                Key('id').begins_with('query-') &
                (Attr('attributes').attribute_type('S') | Attr('attributes.ttl').lt(ttl_expired))
        )
    )

    # Batch delete items
    with ddb_resource.Table(ddb_table_name).batch_writer() as batch:
        for item in response.get('Items', []):
            batch.delete_item(Key={'id': item['id']})



def cache_query(query, response):
    query_hash = get_hash_of_query(query)
    timestamp = int(datetime.now().timestamp())
    ttl = timestamp + (12 * 60 * 60)  # 12 hours from now
    # ttl doesn't work; do it manually
    # we have to basically "hide" this in the db
    ddb_resource.Table(ddb_table_name).put_item(Item={
        'id': f"query-{query_hash}",
        'attributes': {
            "query": query,
            "response": response,
            "timestamp": timestamp,
            "ttl": ttl
        },
    })

    # try:
    #     ddb_resource.update_time_to_live(
    #         TableName=ddb_table_name,
    #         TimeToLiveSpecification={
    #             'Enabled': True,
    #             'AttributeName': "ttl"
    #         }
    #     )
    # except Exception as e:
    #     logger.error(e, exc_info=True)
    #     capture_exception(e)


def get_cached_query(query):
    query_hash = get_hash_of_query(query)
    response = ddb_resource.Table(ddb_table_name).get_item(Key={'id': f"query-{query_hash}"})
    if response.get("Item"):
        # check ttl
        item = response["Item"]
        attributes = item.get("attributes", {})
        # attributes used to be a string, so we have to check
        if isinstance(attributes, str):
            attributes = json.loads(attributes)
        if "ttl" in attributes:
            ttl = attributes["ttl"]
            if int(datetime.now().timestamp()) < ttl:
                return attributes["response"]
            else:
                # delete the item
                ddb_resource.Table(ddb_table_name).delete_item(Key={'id': query_hash})
    return None


def get_perplexity_response(query):
    # Check cache first
    try:
        cached_response = get_cached_query(query)
    except Exception as e:
        logger.error(e, exc_info=True)
        capture_exception(e)
        cached_response = None
    if cached_response:
        return cached_response

    # If not in cache, query Perplexity API
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": query,
        },
    ]
    response = client.chat.completions.create(
        model=MODEL_USED,
        messages=messages,
    )
    answer = response.choices[0].message.content

    # Cache the response
    try:
        cache_query(query, answer)
    except Exception as e:
        logger.error(e, exc_info=True)
        capture_exception(e)

    return answer


def get_subscription_state(handler_input):
    # type: (HandlerInput) -> bool

    user_id = handler_input.request_envelope.session.user.user_id
    locale = get_locale(handler_input)
    try:
        response = handler_input.service_client_factory.get_monetization_service().get_in_skill_products(
            user_id=user_id,
            stage="Live",
            accept_language=locale
        )
        products = response.in_skill_products
        for product in products:
            if product.product_id == SUBSCRIPTION_ID:
                entitlement_state = product.entitled
                if entitlement_state == EntitledState.ENTITLED:
                    return True
    except Exception as e:
        logger.error(e, exc_info=True)
        capture_exception(e)
    return False


class PurchaseFeedbackIntentHandler(AbstractRequestHandler):
    """Allows the user to check what their last pucahsed item was"""

    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_intent_name("PurchaseFeedbackIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response

        attr_manager = handler_input.attributes_manager
        user_attributes = attr_manager.persistent_attributes
        last_purchased_item = user_attributes.get("last_purchased_item", None)

        if last_purchased_item == SMALL_PACK_ID:
            speak_output = get_text(handler_input, "SMALL_QUERY_PACK_FEEDBACK")
        elif last_purchased_item == MEDIUM_PACK_ID:
            speak_output = get_text(handler_input, "MEDIUM_QUERY_PACK_FEEDBACK")
        elif last_purchased_item == LARGE_PACK_ID:
            speak_output = get_text(handler_input, "LARGE_QUERY_PACK_FEEDBACK")
        elif last_purchased_item == SUBSCRIPTION_ID:
            speak_output = get_text(handler_input, "UNLIMITED_QUERY_PACK_FEEDBACK")
        else:
            speak_output = get_text(handler_input, "NO_PURCHASED_ITEM")

        speak_output = speak_output.format(remaining_queries=user_attributes.get("remaining_queries", DEFAULT_QUERIES))

        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )


class CancelSubscriptionIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool

        return ask_utils.is_intent_name("CancelSubscriptionIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response

        speak_output = get_text(handler_input, "REFER_CANCELLATION")

        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )


class QueryIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("QueryIntent")(handler_input)

    def handle(self, handler_input):
        locale = get_locale(handler_input)

        query = handler_input.request_envelope.request.intent.slots["query"].value
        user_id = handler_input.request_envelope.session.user.user_id

        # Check if user has available queries
        attr_manager = handler_input.attributes_manager
        user_attributes = attr_manager.persistent_attributes
        remaining_queries = user_attributes.get("remaining_queries", DEFAULT_QUERIES)

        try:
            cant_use = remaining_queries <= 0
        except Exception as e:
            cant_use = True
            logger.error(e, exc_info=True)
            capture_exception(e)

        if cant_use:
            # check to see if the user has a subscription
            cant_use = get_subscription_state(handler_input)

        if cant_use:
            speak_output = get_text(handler_input, "QUERY_LIMIT_REACHED")
            return (
                handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
            )

        # Process query
        response_text = get_perplexity_response(query)

        # Update remaining queries
        remaining_queries -= 1
        user_attributes["remaining_queries"] = remaining_queries
        attr_manager.persistent_attributes = user_attributes
        attr_manager.save_persistent_attributes()

        # we need to strip markdown
        sanitized_response = strip_markdown_and_newlines(response_text)

        if contains_mature_content(sanitized_response):
            speak_output = get_text(handler_input, "EXPLICIT_CONTENT")
            return (
                handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
            )
        
        if contains_personal_info(sanitized_response):
            speak_output = get_text(handler_input, "PERSONAL_CONTENT")
            return (
                handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
            )

        return (
            handler_input.response_builder
            .speak(f"According to perplexity.ai, {sanitized_response}")
            .ask("Is there anything else you'd like to know?")
            .response
        )


class QueryLimitIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("QueryLimitIntent")(handler_input)

    def handle(self, handler_input):
        attr_manager = handler_input.attributes_manager
        user_attributes = attr_manager.persistent_attributes
        remaining_queries = user_attributes.get("remaining_queries", DEFAULT_QUERIES)
        subscription_state = get_subscription_state(handler_input)

        if subscription_state:
            speak_output = get_text(handler_input, "PREMIUM_OWNED")
        else:
            speak_output = get_text(handler_input, "REMAINING_QUERIES").format(remaining_queries=remaining_queries)
        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )


class BuyQueryPackIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("BuyQueryPackIntent")(handler_input)

    def handle(self, handler_input):
        speak_output = get_text(handler_input, "BUY_QUERIES_PROMPT")
        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )


class BuyProductHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("BuyProductIntent")(handler_input)

    def handle(self, handler_input):
        # product type entry id is either medium_pack, large_pack, or small_pack, or unlimited_pack
        product_id = \
            handler_input.request_envelope.request.intent.slots["product"].resolutions.resolutions_per_authority[
                0].values[
                0].value.id

        # no match on unknown python version
        if product_id == "small_pack":
            amazon_product_id = SMALL_PACK_ID
        elif product_id == "medium_pack":
            amazon_product_id = MEDIUM_PACK_ID
        elif product_id == "large_pack":
            amazon_product_id = LARGE_PACK_ID
        elif product_id == "unlimited_pack":
            amazon_product_id = SUBSCRIPTION_ID
        else:
            raise ValueError("Invalid product ID")

        current_subscription = get_subscription_state(handler_input)

        if current_subscription:
            speak_output = get_text(handler_input, "PREMIUM_OWNED")
            return (
                handler_input.response_builder
                .speak(speak_output)
                .ask(speak_output)
                .response
            )

        return handler_input.response_builder.add_directive(
            SendRequestDirective(
                name="Buy",
                payload={
                    "InSkillProduct": {
                        "productId": amazon_product_id,
                    },
                },
                token="correlationToken"
            )
        ).response


class PurchaseHistoryHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return (ask_utils.is_request_type("Connections.Response")(handler_input) and
                handler_input.request_envelope.request.name == "Buy")

    def handle(self, handler_input):
        purchase_result = handler_input.request_envelope.request.payload.get('purchaseResult')

        if purchase_result == PurchaseResult.ACCEPTED.value:
            speak_output = get_text(handler_input, "PURCHASE_SUCCESSFUL")
            # Update user's query count based on the purchased product
            self.update_query_count(handler_input)
        else:
            speak_output = get_text(handler_input, "PURCHASE_FAILED")

        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )

    def update_query_count(self, handler_input):
        attr_manager = handler_input.attributes_manager
        user_attributes = attr_manager.persistent_attributes
        product_id = handler_input.request_envelope.request.payload.get('productId')

        if product_id == SMALL_PACK_ID:
            user_attributes["remaining_queries"] = user_attributes.get("remaining_queries",
                                                                       DEFAULT_QUERIES) + SMALL_PACK_QUERIES
        elif product_id == MEDIUM_PACK_ID:
            user_attributes["remaining_queries"] = user_attributes.get("remaining_queries",
                                                                       DEFAULT_QUERIES) + MEDIUM_PACK_QUERIES
        elif product_id == LARGE_PACK_ID:
            user_attributes["remaining_queries"] = user_attributes.get("remaining_queries",
                                                                       DEFAULT_QUERIES) + LARGE_PACK_QUERIES

        # store the last purchased item
        user_attributes["last_purchased_item"] = product_id

        attr_manager.persistent_attributes = user_attributes
        attr_manager.save_persistent_attributes()


class HelpIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.HelpIntent")(handler_input)

    def handle(self, handler_input):
        attr_manager = handler_input.attributes_manager
        user_attributes = attr_manager.persistent_attributes
        remaining_queries = user_attributes.get("remaining_queries", DEFAULT_QUERIES)

        speak_output = get_text(handler_input, "HELP").format(remaining_queries=remaining_queries)
        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        speak_output = get_text(handler_input, "GOODBYE")
        return (
            handler_input.response_builder
            .speak(speak_output)
            .response
        )


class FallbackIntentHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_intent_name("AMAZON.FallbackIntent")(handler_input)

    def handle(self, handler_input):
        speak_output = get_text(handler_input, "FALLBACK")
        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )


class SessionEndedRequestHandler(AbstractRequestHandler):
    def can_handle(self, handler_input):
        return ask_utils.is_request_type("SessionEndedRequest")(handler_input)

    def handle(self, handler_input):
        return handler_input.response_builder.response


class CatchAllExceptionHandler(AbstractExceptionHandler):
    def can_handle(self, handler_input, exception):
        return True

    def handle(self, handler_input, exception):
        logger.error(exception, exc_info=True)
        speak_output = get_text(handler_input, "ERROR")
        capture_exception(exception)
        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )


# Main Skill Builder
sb = CustomSkillBuilder(persistence_adapter=dynamodb_adapter)

# Adding handlers
sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(PurchaseFeedbackIntentHandler())
sb.add_request_handler(CancelSubscriptionIntentHandler())
sb.add_request_handler(QueryIntentHandler())
sb.add_request_handler(QueryLimitIntentHandler())
sb.add_request_handler(BuyQueryPackIntentHandler())
sb.add_request_handler(BuyProductHandler())
sb.add_request_handler(PurchaseHistoryHandler())
sb.add_request_handler(HelpIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_request_handler(FallbackIntentHandler())
sb.add_request_handler(SessionEndedRequestHandler())

# Adding exception handlers
sb.add_exception_handler(CatchAllExceptionHandler())

# Lambda handler
skill_builder_lambda_handler = sb.lambda_handler()

_T = TypeVar("_T")


def lambda_handler(event, context):
    # type: (str, _T) -> dict[str, _T]

    try:
        try:
            clean_ddb_table()
        except Exception as e:
            logger.error(e, exc_info=True)
            capture_exception(e)
        with Hub.current.start_transaction(op="lambda_handler", name="lambda_handler"):
            return skill_builder_lambda_handler(event, context)
    finally:
        flush()
