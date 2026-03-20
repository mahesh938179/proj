from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from accounts.models import Profile, Role

class Command(BaseCommand):
    help = 'Sets the role for a user'

    def add_arguments(self, parser):
        parser.add_argument('username', type=str, help='The username of the user')
        parser.add_argument('role', type=str, choices=Role.values, help='The role to assign')

    def handle(self, *args, **options):
        username = options['username']
        role = options['role']
        
        try:
            user = User.objects.get(username=username)
            # Use get_or_create in case the user was created before the signal was added
            profile, created = Profile.objects.get_or_create(user=user)
            profile.role = role
            profile.save()
            self.stdout.write(self.style.SUCCESS(f'Successfully set role {role} for user {username}'))
        except User.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'User {username} does not exist'))
