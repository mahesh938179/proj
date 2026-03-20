from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class Role(models.TextChoices):
    USER = 'user', 'User'
    ADMIN = 'admin', 'Admin'
    SUPERADMIN = 'superadmin', 'SuperAdmin'

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(
        max_length=20,
        choices=Role.choices,
        default=Role.USER
    )

    def __str__(self):
        return f"{self.user.username} - {self.get_role_display()}"

    def is_superadmin(self):
        return self.role == Role.SUPERADMIN

    def is_admin_or_higher(self):
        return self.role in [Role.ADMIN, Role.SUPERADMIN]

class DeletionRequest(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='deletion_request')
    requested_at = models.DateTimeField(auto_now_add=True)
    is_approved = models.BooleanField(default=False)
    is_rejected = models.BooleanField(default=False)

    def __str__(self):
        return f"Deletion request for {user.username}"

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        Profile.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()
